import os 
import requests
import json
import ast 
import re

import numpy as np
import pandas as pd

import time
from datetime import datetime

from IPython.display import clear_output

from matplotlib.ticker import FuncFormatter
from matplotlib.path import Path
import matplotlib.path as mpath

import pyarrow as pa
import pyarrow.parquet as pq

pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter

#from PIL import Image
#from io import BytesIO

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.coordinates import GeocentricTrueEcliptic

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde

from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astroquery.ipac.irsa import Irsa
#from astroquery.sdss import SDSS
from astroquery.mast import Catalogs

import io
from io import StringIO
from dl import authClient, queryClient
import pandas as pd


#DOWNLOAD DATA
def process_fits(path):
    hdul = fits.open(path)
    data_hdu = hdul[1].data

    table = Table(data_hdu)

    names_1d = [col for col in table.colnames if table[col].ndim == 1]
    table_1d = table[names_1d]

    df = table_1d.to_pandas()
    hdul.close()
    return df


#CHANDRA
#https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=IX/70/csc21mas&-out.max=50&-out.form=HTML%20Table&-out.add=_r&-out.add=_RAJ,_DEJ&-sort=_r&-oc.form=sexa
def search_chandra(ra, dec, radius_deg):
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(
                                    SkyCoord(ra=ra, 
                                             dec=dec, 
                                             unit=(u.deg, u.deg), 
                                             frame='icrs'),
                                    radius=radius_deg * u.deg,
                                    catalog="IX/70/csc21mas"
                                )
    try:
        table = result[0]
        df = table.to_pandas()
        df = df.rename(columns={"r0": "err", 
                                "RAICRS": "ra", 
                                "DEICRS": "dec"
                                })
    except:
        df = pd.DataFrame()

# search for WISE objects in area
def search_panstarrs(ra, dec, radius):
    panstarrs_results = Catalogs.query_region(f"{ra} {dec}",radius=radius*u.deg,catalog="Panstarrs")
    panstarrs_results_df = panstarrs_results.to_pandas()
    
    return panstarrs_results_df

def search_wise(ra, dec, radius):
    
    wise_results = Irsa.query_region(SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'), catalog='allwise_p3as_psd', spatial='Cone', radius=radius*u.deg)  
    wise_results_df = wise_results.to_pandas()
    
    return wise_results_df

#  search for Gaia DR3 objects in area
def search_gaia(ra, dec, radius):
    
    job = Gaia.cone_search_async(SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'), radius=radius*u.deg)  
    gaia_results = job.get_results()
    gaia_df =  gaia_results.to_pandas()

    return gaia_df

#Searches ZTF DF22 catalogue for the objects
def search_ztf(ra, dec, radius, additional_processing = True):
    
    result = Irsa.query_region(SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'),catalog='ztf_objects_dr22', radius=radius*u.deg)
    ztf_df = result.to_pandas()

    if additional_processing == True:
        ztf_grouping(ztf_df)
        #get_lighturves_for_ZTF(ztf_df)
        add_lightcurves_to_ztf_df_with_web(ztf_df)

    return ztf_df

def add_lightcurves_to_ztf_df_with_web(df):

    def get_ligthcurve_from_web(oid):
        base_url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?"
        params = [
            f"ID={oid}",
            "FORMAT=ipac_table",
            "COLLECTION=ztf_dr22" 
        ]
        query_url = base_url + "&".join(params)

        response = requests.get(query_url)
        text = response.text

        lines = text.split('\n')

        data_lines = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit(): 
                data_lines.append(line)

        filtered_text = "\n".join(data_lines)

        col_names = [
            "oid", "expid", "hjd", "mjd", "mag", "magerr", "catflags", "filtercode",
            "ra", "dec", "chi", "sharp", "filefracday", "field", "ccdid", "qid",
            "limitmag", "magzp", "magzprms", "clrcoeff", "clrcounc", "exptime",
            "airmass", "programid"
        ]

        df = pd.read_csv(io.StringIO(filtered_text), 
                        delim_whitespace=True, 
                        names=col_names)

        mjd = df['mjd'].tolist()
        mag = df['mag'].tolist()
        magerr = df['magerr'].tolist()

        return mjd, mag, magerr

    df['hmjd'] = None
    df['mag'] = None
    df['magerr'] = None

    for index, row in df.iterrows():
        oid = row['oid']
        mjd, mag, magerr  =  get_ligthcurve_from_web(oid)
        
        df.at[index, 'hmjd'] = mjd
        df.at[index, 'mag'] = mag
        df.at[index, 'magerr'] = magerr
    
    return df


# Assign ZTF to a group based on the distance between ZTF's 
def ztf_grouping(ztf_closest, all_to_one=False, max_distance_deg = 1.6 / 3600 ):

    if len(ztf_closest) <= 1:
        ztf_closest['group'] = 1
        return ztf_closest

    ra = ztf_closest['ra'].values
    dec = ztf_closest['dec'].values

    dist_matrix = pdist(np.column_stack([ra, dec]), metric='euclidean')

    cl = linkage(dist_matrix, method='complete')
    clusters = fcluster(cl, t=max_distance_deg, criterion='distance')

    ztf_closest['group'] = clusters

    return ztf_closest

#NVSS  Radio catalog
def search_for_nvss(ra_xray,dec_xray,err_xray):

    coord = SkyCoord(ra_xray*u.deg, dec_xray*u.deg, frame="icrs")


    result = Vizier.query_region(
                                coord, 
                                radius=err_xray*u.deg,           
                                catalog="VIII/65/nvss" #catalogs_to_query 
                                ) 

    if "VIII/65/nvss" in result.keys():
        nvss_table = result["VIII/65/nvss"]
        df = nvss_table.to_pandas()
        return df
    else:
        return pd.DataFrame()

#MORXV2 Radio catalog
def search_for_morxv2(ra_xray,dec_xray,err_xray):

    coord = SkyCoord(ra_xray*u.deg, dec_xray*u.deg, frame="icrs")

    result = Vizier.query_region(
                                coord, 
                                radius=err_xray*u.deg,           
                                catalog="V/158/morxv2" #catalogs_to_query 
                                ) 

    if "V/158/morxv2" in result.keys():
        nvss_table = result["V/158/morxv2"]
        df = nvss_table.to_pandas()
        return df
    else:
        return pd.DataFrame()

def search_for_legacy(ra_xray,dec_xray,err_xray):
    query = f"""
    SELECT * 
    FROM ls_dr10.tractor
    WHERE q3c_radial_query(ra, dec, {ra_xray}, {dec_xray}, {err_xray})
    """ 

    res_str = queryClient.query(
        token=token,
        sql=query,
        outfmt='csv'   
    )

    res_df = pd.read_csv(StringIO(res_str))
    return res_df
    
 #CALCULATIONS 
def add_separation_columns (ra_ref, dec_ref, df):
    target = SkyCoord(ra=ra_ref*u.deg, dec=dec_ref*u.deg, frame='icrs')

    coords = SkyCoord(
        ra  = df['ra'].values  * u.deg,
        dec = df['dec'].values * u.deg,
        frame='icrs'
    )

    # angular separation
    sep = coords.separation(target)        

    df['separation (deg)']   = sep.degree
    df['separation (arcsec)'] = sep.arcsec

    return df

def circular_search_ufo_and_cats(ufo_df,xray_df, mode = 'just_count'):
    
    xray_coords = SkyCoord(
                            ra=xray_df["ra"].values * u.deg,
                            dec=xray_df["dec"].values * u.deg
                            )

    ufo_matches = []

    for idx, ufo_row in ufo_df.iterrows():
        ufo_coord = SkyCoord(
                            ra=ufo_row["ra"] * u.deg,
                            dec=ufo_row["dec"] * u.deg
                            )
        
        sep = ufo_coord.separation(xray_coords)
        
        within_radius = sep <= (ufo_row["semi_major_3sigma(deg)"] * u.deg)
        
        matched_xray_objects = xray_df[within_radius].copy()

        ufo_df.loc[idx, 'count_search'] = ufo_row['count_search'] + len(matched_xray_objects)

        if mode == 'return_df':
            xray_df['UFO name'] = ufo_row['#name']
            ufo_matches.append(matched_xray_objects)          

    if mode == 'return_df':
        ufo_matches = pd.concat(ufo_matches, ignore_index=True)  
        return ufo_df, ufo_matches

    if mode == 'just_count':
        return ufo_df


def check_falling_in_wise_countours(x,y,dx,dy):

    def point_in_contours(x, y, dx, dy, contours_65, contours_95):

        inside_65 = False
        inside_95 = False
        
        error_x = np.linspace(x - dx, x + dx, 10)
        error_y = np.linspace(y - dy, y + dy, 10)
        error_grid = np.array(np.meshgrid(error_x, error_y)).T.reshape(-1, 2)  

        for contour in contours_65:
            path = Path(contour)  
            if path.contains_point((x, y)) or np.any(path.contains_points(error_grid)):
                inside_65 = True

        for contour in contours_95:
            path = Path(contour)  
            if path.contains_point((x, y)) or np.any(path.contains_points(error_grid)):
                inside_95 = True

        return inside_65, inside_95

    json_filename = "blasars_pulsars_contours.json"

    with open(json_filename, "r") as json_file:
        loaded_data = json.load(json_file)

    loaded_blasars_65 = [np.array(contour) for contour in loaded_data["blasars_65"]]
    loaded_blasars_95 = [np.array(contour) for contour in loaded_data["blasars_95"]]

    loaded_pulsars_65 = [np.array(contour) for contour in loaded_data["pulsars_65"]]
    loaded_pulsars_95 = [np.array(contour) for contour in loaded_data["pulsars_95"]]

    inside_blasars_65, inside_blasars_95 = point_in_contours(x, y, dx, dy, loaded_blasars_65, loaded_blasars_95)
    inside_pulsars_65, inside_pulsars_95 = point_in_contours(x, y, dx, dy, loaded_pulsars_65, loaded_pulsars_95)
    
    return inside_blasars_65, inside_blasars_95, inside_pulsars_65, inside_pulsars_95

#plot 3 sigma region of UFO 
def deg_to_dms_formatter(x, pos):
    deg = int(x)
    arcmin = int((x - deg) * 60)
    arcsec = (x - deg - arcmin / 60) * 3600
    return f"{deg}°{arcmin}′{arcsec:.2f}″"

def does_wise_in_the_box(x_center, y_center, x_err, y_err):
    x23_y12 = {'x': [1.52, 2.59, 3.30, 2.01, 1.52], 'y': [0.51, 1.20, 1.17, 0.37, 0.51]}
    region_path = mpath.Path(np.column_stack((x23_y12['x'], x23_y12['y'])))

    bbox_points = [
        (x_center - x_err, y_center - y_err),
        (x_center + x_err, y_center - y_err),
        (x_center + x_err, y_center + y_err),
        (x_center - x_err, y_center + y_err)
    ]

    # Check if any part of the bounding box intersects the region
    if any(region_path.contains_point(point) for point in bbox_points):
        return True
    else:
        return False

#Check if X-ray fits in the 3sigma ellips
def xray_inside_ellipse(potential_objects, center_ra, center_dec, semi_major, semi_minor, angle_deg, object_name):
    
    associated_xrays = potential_objects[potential_objects['name'] == object_name]

    if associated_xrays.empty:
        return pd.DataFrame(columns=potential_objects.columns)

    angle_rad = np.radians(angle_deg)
    
    delta_ra = associated_xrays['RA'] - center_ra
    delta_dec = associated_xrays['Dec'] - center_dec

    x_rot = delta_ra * np.cos(angle_rad) + delta_dec * np.sin(angle_rad)
    y_rot = -delta_ra * np.sin(angle_rad) + delta_dec * np.cos(angle_rad)
    
    distance = (x_rot ** 2) / (semi_major ** 2) + (y_rot ** 2) / (semi_minor ** 2)

    inside_ellipse = associated_xrays[distance <= 1]
    return inside_ellipse 


def classify_ztf_groups(df, step_sig =2, sigma_clip=5.0):
    results = {}
    for group_id, group_data in df.groupby('group'):
        all_means = []
        parsed_rows = []

        for idx, row in group_data.iterrows():
            try:
                mags = row['mag']
                magerrs = row['magerr']

                if isinstance(mags, str):
                    mags = ast.literal_eval(mags)
                if isinstance(magerrs, str):
                    magerrs = ast.literal_eval(magerrs)

                if len(mags) > 0:
                    mean_mag = np.mean(mags)
                    all_means.append(mean_mag)
                    parsed_rows.append((mags, magerrs))
            except Exception as e:
                print(f"Warning: problem parsing group {group_id}: {e}")

        if all(mean_mag > 20 for mean_mag in all_means) and len(all_means) > 0:
            results[group_id] = 'faint'
            continue

        max_length = -1
        best_row = None

        for mags, magerrs in parsed_rows:
            if len(mags) > max_length:
                max_length = len(mags)
                best_row = (mags, magerrs)

        if best_row is None:
            results[group_id] = 'unk'
            continue

        mags, magerrs = best_row
        mags = np.array(mags)
        magerrs = np.array(magerrs)

        # Clip outliers
        median_mag = np.median(mags)
        std_mag = np.std(mags)
        mask = np.abs(mags - median_mag) < sigma_clip * std_mag

        ignored_points = np.sum(~mask)
        if ignored_points > 0:
            print(f"In group {group_id} ignored {ignored_points} points as outliers.")

        mags = mags[mask]
        magerrs = magerrs[mask]

        if len(mags) < 5:
            results[group_id] = 'unk'
            continue

        std_mag = np.std(mags)
        mean_magerr = np.mean(magerrs)

        if std_mag > step_sig * mean_magerr:
            results[group_id] = 'var'
        elif std_mag < step_sig * mean_magerr:
            results[group_id] = 'flat'

    return results
    
#PLOTTING

#WISE COLOURS 
def plot_3sig_ellips(w2w3, w1w2, color = None, extract_values = False): 
    #extract_values make funct  return countours of 65 and 95 
    #color assigns special color to the countours

    xy = np.vstack([w2w3, w1w2])
    kde = gaussian_kde(xy)
    density = kde(xy)

    density_sorted = np.sort(density)
    cumulative_density = np.cumsum(density_sorted) / np.sum(density_sorted)

    threshold_95 = density_sorted[np.searchsorted(cumulative_density, 0.45)]
    threshold_65 = density_sorted[np.searchsorted(cumulative_density, 0.05)]

    countur_points_number =  1000 #600
    x = np.linspace(w2w3.min(), w2w3.max(), countur_points_number)
    y = np.linspace(w1w2.min(), w1w2.max(), countur_points_number)
    X, Y = np.meshgrid(x, y)
    grid_coords = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(grid_coords).reshape(X.shape)
    
    #plt.contourf(X, Y, Z, levels=50, cmap='Blues')  # Density map
    #plt.colorbar(label='Density')

    if color == None:
        colors = ['orange', 'red']
    else:
        base_color = mcolors.to_rgba(color)  
        light_color = (*base_color[:3], 0.5)  
        intense_color = (*base_color[:3], 1.0)  
        colors = [light_color, intense_color] 

    if extract_values == False:
        plt.contour(X, Y, Z, levels=[threshold_65, threshold_95], colors=colors, linewidths=1.5, linestyles=['--', '-'])

    if extract_values == True:
        cs = plt.contour(
            X, Y, Z,
            levels=[threshold_65, threshold_95],
            colors=colors, linewidths=1.5, linestyles=['--', '-']
        )

        contours_65 = []  
        contours_95 = []  

        for level_idx, level_val in enumerate(cs.levels):
            level_paths = cs.collections[level_idx].get_paths()
            for p in level_paths:
                if np.isclose(level_val, threshold_65):
                    contours_65.append(p.vertices)
                elif np.isclose(level_val, threshold_95):
                    contours_95.append(p.vertices)

        return contours_65, contours_95

#XRAY with GAIA ZTF WISE 
def plot_position(ra_xray, dec_xray, err_xray, inst_xray,object_name, ztf_closest, plot_grouping = True, gaia_sources = pd.DataFrame(),wise_sources = pd.DataFrame()):
        plt.figure(figsize=(10, 6))
        #ZTF's coords

        ztf_ra =  np.array(ztf_closest['ra']) 
        ztf_dec =  np.array(ztf_closest['dec']) 
        #plt.scatter(ztf_ra, ztf_dec, color='black', alpha=0.7, label='ZTF all')

        #ZTF grouping
        groups = ztf_closest['group'].unique()

        if plot_grouping:
            for group in groups:
                    group_indices = ztf_closest['group'] == group
                    group_ra = np.array(ztf_closest[group_indices]['ra'])
                    group_dec = np.array(ztf_closest[group_indices]['dec'])
                    plt.plot(group_ra, group_dec, linestyle='--', alpha=0.7, color='gray')
                    
                    group_center_ra = np.mean(group_ra)
                    group_center_dec = np.mean(group_dec)
                    
                    plt.annotate(f'Group {group}', xy=(group_center_ra, group_center_dec), xytext=(5, 5), 
                            textcoords='offset points', fontsize=10, color='blue')

        #ZTF red green ir
        ztf_red = ztf_closest[ztf_closest['filtercode'] == "zr"]
        plt.scatter(np.array(ztf_red['ra']), np.array(ztf_red['dec']), color='red', label='ZTF: red filter')

        ztf_green = ztf_closest[ztf_closest['filtercode'] == "zg"]
        plt.scatter(np.array(ztf_green['ra']), np.array(ztf_green['dec']), color='green', label='ZTF: green filter')

        ztf_infrared = ztf_closest[ztf_closest['filtercode'] == "zi"]
        plt.scatter(np.array(ztf_infrared['ra']), np.array(ztf_infrared['dec']), color='purple', label='ZTF: infrared filter')

        problematic_ztf_plot = ztf_closest[ztf_closest['hmjd'].apply(lambda x: x is None or isinstance(x, (float, int)) or (isinstance(x, np.ndarray) and x.size == 1) or (isinstance(x, np.ndarray) and x.size == 2))] 
        plt.scatter(np.array(problematic_ztf_plot['ra']), np.array(problematic_ztf_plot['dec']),marker='x', color='black', label='ZTF: problematic')
        
        if not gaia_sources.empty:
            colors = plt.cm.Blues(np.linspace(0.3, 1, len(gaia_sources))) 
            for (index_other_sets, row_other_sets), color in zip(gaia_sources.iterrows(), colors):
                    plt.errorbar(
                            row_other_sets['ra'], 
                            row_other_sets['dec'], 
                            xerr=row_other_sets['ra_error']/3600/1000, 
                            yerr=row_other_sets['dec_error']/3600/1000, 
                            marker='*',
                            color=color,  # Use the current shade of blue
                            label=f'{row_other_sets["DESIGNATION"]}', 
                            markersize=10
                    )

        if not wise_sources.empty:
            colors = plt.cm.Oranges(np.linspace(0.3, 1, len(wise_sources))) 
            for (index_other_sets, row_other_sets), color in zip(wise_sources.iterrows(), colors):
                    plt.errorbar(
                            row_other_sets['ra'], 
                            row_other_sets['dec'], 
                            xerr=row_other_sets['sigra']/3600/1000, 
                            yerr=row_other_sets['sigdec']/3600/1000, 
                            marker='*',
                            color=color,  # Use the current shade of blue
                            label=f'{row_other_sets["designation"]}', 
                            markersize=10
                    )



        plt.errorbar(ra_xray, 
                     dec_xray, 
                     xerr=err_xray/3600, 
                     yerr=err_xray/3600, 
                     color='black', 
                     label=f'X-ray Object: {inst_xray}',
                     markersize=10)

        #ADJUST
        radii_= 1.2 * err_xray/3600 
        
        plt.xlim(ra_xray - radii_, ra_xray + radii_) 
        plt.ylim(dec_xray - radii_, dec_xray + radii_)  


        arcsec_to_deg = 10 / 3600  
        scale_x_start = ra_xray - 0.5 * arcsec_to_deg
        scale_x_end = ra_xray + 0.5 * arcsec_to_deg
        scale_y = dec_xray - 1.15 * err_xray / 3600 

        plt.hlines(
        y=scale_y, 
        xmin=scale_x_start, 
        xmax=scale_x_end, 
        colors='red', 
        linestyles='solid', 
        label='10 arcsec scale'
        )

        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(deg_to_dms_formatter))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(deg_to_dms_formatter))
        plt.xticks(rotation=15)

        plt.legend(frameon=False,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"X-ray and ZTF objects for {object_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{object_name}_xray_positions.png',dpi=300)
        plt.show()


#ZTF GROUPS
def plot_ztf_groups(df, object_name,save_the_image =False, img_size = 'small',save_the_lightcurves =False, year_lines=True):
    grouped = df.groupby('group')
    n_groups = len(grouped)  # Number of groups

    # Create a single figure with subplots for each group
    if img_size == 'big':
        fig, axes = plt.subplots(n_groups, 1, figsize=(15, 6 * n_groups), sharex=True)
        markersize_points = 6
    
    if img_size == 'small':
        fig, axes = plt.subplots(n_groups, 1, figsize=(6, 3 * n_groups), sharex=True)
        markersize_points = 5
        
    fig.suptitle(f"{object_name}", fontsize=10)

    if year_lines:
        flat_hmjd = np.concatenate(df['hmjd'].to_numpy())
        years     = np.arange(
            Time(flat_hmjd.min(), format='mjd').datetime.year,
            Time(flat_hmjd.max(), format='mjd').datetime.year + 1
        )
        
        year_mjd  = [Time(f'{y}-01-01T00:00:00', format='isot').mjd for y in years]

    if n_groups == 1:
        axes = [axes]

    for (group_name, group_data), ax in zip(grouped, axes):
        ax.set_title(f"Group {group_name}", fontsize=7)
        ax.set_xlabel("HMJD")
        ax.set_ylabel("Magnitude")

        filter_grouped = group_data.groupby('filtercode')
        
        color_map = {'zr': 'red', 
                     'zg': 'green', 
                     'zi': 'purple'} 


        for filter_code, filter_data in filter_grouped:
            #print(filter_code)
            all_hmjd = []
            all_mag = []
            all_magerr = []

            for _, row in filter_data.iterrows():
                all_hmjd.extend(row['hmjd'])
                all_mag.extend(row['mag'])
                all_magerr.extend(row['magerr'])

            color = color_map.get(filter_code, 'black') 
            
            all_hmjd_array = np.array(all_mag)
            
            if np.mean(all_hmjd_array)+np.std(all_hmjd_array) >= 20:
                ax.axhline(20, linestyle='dashed', alpha=0.5)
                ax.axhline(21, linestyle='solid', alpha=0.5)

            '''
            if (save_the_lightcurves == True) & (save_the_image ==  True):
                 dir_path = os.join.path(os.getcwd(),f'{object_name}')
                 if not os.path.exists(dir_path):
                    os.mkdir(dir_path)

                 plt.savefig(os.join.path(dir_path,f'{object_name}_ZTF_lightcurves.png'), dpi=300)
            '''
                 
            ax.errorbar(all_hmjd, all_mag, yerr=all_magerr, fmt='o', label=f"Filter {filter_code}", ecolor='grey', capsize=1, markersize=markersize_points, color=color)
        
        if year_lines:
            for mjd in year_mjd:
                ax.axvline(mjd, ls=':', lw=1, color='grey', alpha=0.6)  

        #ax.legend()
        ax.invert_yaxis() 


    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle
    
    if (save_the_lightcurves == False) & (save_the_image ==  True):
        plt.savefig(f'{object_name}_ZTF_lightcurves.png', dpi=300)
    plt.show()

    
def plot_sky_in_galactic(df,text,colour='blue',size=2,df_2=pd.DataFrame(),colour_2='red'):
    try:
        skycoord = SkyCoord(
            ra=df['RAJ2000'].values * u.deg,
            dec=df['DEJ2000'].values * u.deg,
            frame='icrs'
        )
    except:
        skycoord = SkyCoord(
            ra=df['ra'].values * u.deg,
            dec=df['dec'].values * u.deg,
            frame='icrs'
        )
    l = skycoord.galactic.l.wrap_at(180 * u.deg).radian  
    b = skycoord.galactic.b.radian

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='mollweide')

    ax.scatter(l, b, s=size, color=colour, alpha=0.7)

    if not df_2.empty:

        try:
            skycoord = SkyCoord(
                ra=df_2['RAJ2000'].values * u.deg,
                dec=df_2['DEJ2000'].values * u.deg,
                frame='icrs'
            )
        except:
            skycoord = SkyCoord(
                ra=df_2['ra'].values * u.deg,
                dec=df_2['dec'].values * u.deg,
                frame='icrs'
            )
        l = skycoord.galactic.l.wrap_at(180 * u.deg).radian  
        b = skycoord.galactic.b.radian

        ax.scatter(l, b, s=size, color=colour_2, alpha=0.7)

    ax.grid(True)
    
    if text:
        #plt.title(f"{text}")
        plt.savefig(f"{text}.png", dpi=300)
    plt.show()

def plot_ufo(ra_c,dec_c,smj,smi,ang,label_info,df):
    theta = np.radians(ang)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    fig, ax = plt.subplots(figsize=(10, 8))  

    angles = np.linspace(0, 2*np.pi, 100)
    x_ell = smj * np.cos(angles)
    y_ell = smi * np.sin(angles)

    X_ell =  x_ell*cos_t - y_ell*sin_t
    Y_ell =  x_ell*sin_t + y_ell*cos_t

    ra_ellipse  = ra_c  + (X_ell / np.cos(np.radians(dec_c)))
    dec_ellipse = dec_c +  Y_ell

    ax.plot(ra_ellipse, dec_ellipse)
    ax.plot(ra_c, dec_c, 'o')

    ax.set_title(label_info)
    ax.plot(df["ra"], df["dec"], '.')

    ax.invert_xaxis()

    def ra_formatter(x, _):
        angle = Angle(x, unit=u.deg)
        return angle.to_string(unit=u.hour, 
                            sep=':', 
                            precision=0, 
                            pad=True)

    def dec_formatter(y, _):
        angle = Angle(y, unit=u.deg)
        return angle.to_string(unit=u.deg, 
                            sep=':', 
                            precision=0, 
                            alwayssign=True, 
                            pad=True)

    ax.xaxis.set_major_formatter(FuncFormatter(ra_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(dec_formatter))

    plt.tight_layout()
    plt.show()
