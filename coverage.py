import pandas as pd  
import argparse
import json
import os
import time
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
import numpy as np

import fiona
import geopandas as gpd
import pandas as pd
import requests

# to do replace with storage
DATA_DIR = Path(SPINEQ_HOME, "data")
RAW_DIR = Path(DATA_DIR, "raw")
PROCESSED_DIR = Path(DATA_DIR, "processed")

def route_coverage():
    """Compute coverage stats given a list of output areas with sensors.

    Query parameters JSON:
        - sensors: List of OA (oa11cd) with sensors
        - theta: decay rate for coverage measure (default: 500)
        - lad20cd: Local authority codde (default: E08000021)

    Returns:
        dict -- json of OA and overall coverage stats
    """
    args = request.get_json()
    if "sensors" not in args:
        return "No sensors defined", 400
    theta = float(args.get("theta", 500))
    lad20cd = args.get("lad20cd", "E08000021")
    inputs = get_optimisation_inputs(
        lad20cd=lad20cd,
        population_weight=1,
        workplace_weight=1,
        pop_age_groups={
            "pop_total": {"min": 0, "max": 90, "weight": 1},
            "pop_children": {"min": 0, "max": 15, "weight": 1},
            "pop_elderly": {"min": 66, "max": 90, "weight": 1},
        },
        combine=False,
    )
    oa_weight = pd.DataFrame(inputs["oa_weight"], index=inputs["oa11cd"])
    return jsonify(
        calc_coverage(
            lad20cd,
            args.get("sensors"),
            oa_weight=oa_weight,
            theta=theta,
        )
    )


def get_optimisation_inputs(
    lad20cd="E08000021",
    population_weight=1,
    workplace_weight=0,
    pop_age_groups={
        "pop_total": {"min": 0, "max": 90, "weight": 1},
        "pop_children": {"min": 0, "max": 16, "weight": 0},
        "pop_elderly": {"min": 70, "max": 90, "weight": 0},
    },
    combine=True,
):
    """Get input data in format needed for optimisation.

    Keyword Arguments:
        lad20cd, population_weight, workplace_weight, pop_age_groups, combine -- As
        defined in calc_oa_weights (parameters directly passed to that
        function.)

    Returns:
        dict -- Optimisation input data
    """
    centroids = (lad20cd=lad20cd)
    weights = calc_oa_weights(
        lad20cd=lad20cd,
        population_weight=population_weight,
        workplace_weight=workplace_weight,
        pop_age_groups=pop_age_groups,
        combine=combine,
    )
    if type(weights) == pd.DataFrame:
        weights_columns = ["weight_" + name for name in weights.columns]
        weights.columns = weights_columns
    else:
        weights.name = "weights"

    if len(centroids) != len(weights):
        raise ValueError(
            "Lengths of inputs don't match: centroids={}, weights={}".format(
                len(centroids), len(weights)
            )
        )

    centroids = centroids.join(weights)

    oa11cd = centroids.index.values
    oa_x = centroids["x"].values
    oa_y = centroids["y"].values

    if type(weights) == pd.DataFrame:
        oa_weight = {
            name.replace("weight_", ""): centroids[name].values
            for name in weights_columns
        }
    else:
        oa_weight = centroids[weights.name].values

    return {"oa11cd": oa11cd, "oa_x": oa_x, "oa_y": oa_y, "oa_weight": oa_weight}

def calc_coverage(lad20cd, sensors, oa_weight=None, theta=500):
    """Calculate the coverage of a network for arbitrary OA weightings.

    Arguments:
        sensors {list} -- List of OA (oa11cd) with a sensor.
        oa_weight {pd.Series} -- Weight for each OA, pandas series with index
        oa11cd and weights as values. If None weight all OA equally.

    Keyword Arguments:
        theta {int} -- coverage decay rate (default: {500})

    Returns:
        dict -- Coverage stats with keys "total_coverage" and "oa_coverage".
    """
    centroids = get_oa_centroids(lad20cd)

    oa_x = centroids["x"].values
    oa_y = centroids["y"].values
    coverage = coverage_matrix(oa_x, oa_y, theta=theta)

    # only keep coverages due to sites where a sensor is present
    centroids["has_sensor"] = 0
    for oa in sensors:
        centroids.loc[oa, "has_sensor"] = 1
    sensors = centroids["has_sensor"].values
    mask_cov = np.multiply(coverage, sensors[np.newaxis, :])

    # coverage at each site = coverage due to nearest sensor
    oa_coverage = np.max(mask_cov, axis=1)

    # Avg coverage = weighted sum across all points of interest
    if oa_weight is None:
        oa_weight = 1  # equal weights for each OA
    elif isinstance(oa_weight, pd.DataFrame) and len(oa_weight.columns) == 1:
        oa_weight = oa_weight[oa_weight.columns[0]]

    if isinstance(oa_weight, pd.DataFrame):
        overall_coverage = {}
        for obj in oa_weight.columns:
            centroids["weight"] = oa_weight[obj]
            overall_coverage[obj] = total_coverage(
                oa_coverage, centroids["weight"].values
            )
    else:
        centroids["weight"] = oa_weight
        overall_coverage = total_coverage(oa_coverage, centroids["weight"].values)

    oa_coverage = [
        {"oa11cd": oa, "coverage": cov}
        for oa, cov in zip(centroids.index.values, oa_coverage)
    ]
    return {"total_coverage": overall_coverage, "oa_coverage": oa_coverage}

def total_coverage(point_coverage: np.array, point_weights: np.array = None) -> float:
    """Total coverage metric from coverage of each point

    Parameters
    ----------
    point_coverage : np.array
        Coverage provided at each point (due to a sensor network)
    point_weights : np.array
        Weight for each point

    Returns
    -------
    float
        Total coverage (between 0 and 1)
    """
    return np.average(point_coverage, weights=point_weights)


# Getting data
def get_oa_centroids(lad20cd="E08000021"):
    """Get output area population weighted centroids

    Returns:
        pd.DataFrame -- Dataframe with index oa11cd and columns x and y.
    """
    path = Path(PROCESSED_DIR, lad20cd, "centroids.csv")
    if not path.exists():
        extract_la_data(lad20cd)
    return pd.read_csv(path, index_col="oa11cd")\
    
def coverage_matrix(x1, y1, x2=None, y2=None, theta=1):
    """Generate a matrix of coverages for a number of locations

    Arguments:
        x {list-like} -- x coordinate for each location
        y {list-like} -- y coordinate for each location

    Keyword Arguments:
        theta {numeric} -- decay rate (default: {1})

    Returns:
        numpy array -- 2D matrix of coverage at each location i due to a
        sensor placed at another location j.
    """
    distances = distance_matrix(x1, y1, x2=x2, y2=y2)
    return np.exp(-distances / theta)

def distance_matrix(x1, y1, x2=None, y2=None):
    """Generate a matrix of distances between a number of locations. Either
    pairwise distances between all locations in one set of x and y coordinates,
    or pairwise distances between one set of x,y coordinates (x1, y1) and
    another set of coordinates (x2,y2)

    Arguments:
        x1 {list-like} -- x coordinate for each location
        y1 {list-like} -- y coordinate for each location
        x2 {list-like} -- x coordinate for each location
        y2 {list-like} -- y coordinate for each location

    Returns:
        numpy array -- 2D matrix of distance between location i and location j,
        for each i and j.
    """

    coords_1 = np.array([x1, y1]).T

    if x2 is not None and y2 is not None:
        # calculate distances between two sets of coordinates
        coords_2 = np.array([x2, y2]).T

        dist_sq = np.sum(
            (coords_1[:, np.newaxis, :] - coords_2[np.newaxis, :, :]) ** 2, axis=-1
        )

    elif (x2 is None and y2 is not None) or (y2 is None and x2 is not None):
        raise ValueError("x2 and y2 both must be defined or undefined.")

    else:
        # calculate distances distances between points in one set of coordinates
        dist_sq = np.sum(
            (coords_1[:, np.newaxis, :] - coords_1[np.newaxis, :, :]) ** 2, axis=-1
        )

    distances = np.sqrt(dist_sq)

    return distances

def extract_la_data(lad20cd="E08000021", overwrite=False):
    print(f"Extracting data for {lad20cd}...")
    save_dir = Path(PROCESSED_DIR, lad20cd)
    os.makedirs(save_dir, exist_ok=True)

    la = download_la_shape(lad20cd=lad20cd, overwrite=overwrite)
    print("LA shape:", len(la), "rows")

    mappings = download_oa_mappings(overwrite=overwrite)
    oa_in_la = mappings.loc[mappings["lad20cd"] == lad20cd, "oa11cd"]
    print("OA in this LA (mappings):", len(oa_in_la), "rows")

    lad11cd = lad20cd_to_lad11cd(lad20cd, mappings)
    oa = download_oa_shape(lad11cd=lad11cd, overwrite=overwrite)
    print("OA shapes:", len(oa), "rows")

    # centroids
    centroids = download_centroids(overwrite=overwrite)
    centroids = filter_oa(oa_in_la, centroids)
    centroids.to_csv(Path(save_dir, "centroids.csv"), index=False)
    print("Centroids:", len(centroids), "rows")

    # population data
    population_total, population_ages = download_populations(overwrite=overwrite)
    population_total = filter_oa(oa_in_la, population_total)
    population_total.to_csv(Path(save_dir, "population_total.csv"), index=False)
    print("Total Population:", len(population_total), "rows")

    population_ages = columns_to_lowercase(population_ages)
    population_ages = filter_oa(oa_in_la, population_ages)
    population_ages.to_csv(Path(save_dir, "population_ages.csv"), index=False)
    print("Population by Age:", len(population_ages), "rows")

    # workplace
    workplace = download_workplace(overwrite=overwrite)
    workplace = filter_oa(oa_in_la, workplace)
    workplace.to_csv(Path(save_dir, "workplace.csv"), index=False)
    print("Place of Work:", len(workplace), "rows")

    if not (
        len(oa) == len(centroids)
        and len(oa) == len(population_total)
        and len(oa) == len(population_ages)
        and len(oa) == len(workplace)
    ):
        warnings.warn("Lengths of processed data don't match, optimisation will fail!")

    process_uo_sensors(lad20cd=lad20cd, overwrite=overwrite)


def download_la_shape(lad20cd="E08000021", overwrite=False):
    save_path = Path(PROCESSED_DIR, lad20cd, "la_shape", "la.shp")
    if os.path.exists(save_path) and not overwrite:
        return gpd.read_file(save_path)
    os.makedirs(save_path.parent, exist_ok=True)

    # From https://geoportal.statistics.gov.uk/datasets/
    #          ons::local-authority-districts-december-2020-uk-bgc/about
    base = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "Local_Authority_Districts_December_2020_UK_BGC/FeatureServer/0"
    )
    query = (
        f"query?where=LAD20CD%20%3D%20%27{lad20cd}%27&outFields=*&outSR=27700&f=json"
    )
    url = f"{base}/{query}"
    la = query_ons_records(url, save_path=None)
    la = columns_to_lowercase(la)
    la = la[["geometry", "lad20cd", "lad20nm"]]
    la.to_file(save_path)
    return la

def download_oa_shape(lad11cd="E08000021", lad20cd=None, overwrite=False):
    if isinstance(lad11cd, str):
        lad11cd = [lad11cd]
    if lad20cd is None:
        lad20cd = lad11cd_to_lad20cd(lad11cd[0])[0]

    save_path = Path(PROCESSED_DIR, lad20cd, "oa_shape", "oa.shp")
    if os.path.exists(save_path) and not overwrite:
        return gpd.read_file(save_path)
    os.makedirs(save_path.parent, exist_ok=True)

    oa = []
    for la in lad11cd:
        # From https://geoportal.statistics.gov.uk/datasets/
        #             ons::output-areas-december-2011-boundaries-ew-bgc-1/about
        url = (
            "https://ons-inspire.esriuk.com/arcgis/rest/services/Census_Boundaries/"
            "Output_Area_December_2011_Boundaries/FeatureServer/2/query?"
            f"where=lad11cd%20%3D%20'{la}'&outFields=*&outSR=27700&f=json"
        )
        oa.append(query_ons_records(url, save_path=None))

    oa = pd.concat(oa)
    oa = columns_to_lowercase(oa)
    oa = oa[["oa11cd", "geometry"]]
    oa.to_file(save_path)
    return oa


def download_oa_mappings(overwrite=False):
    save_path = Path(RAW_DIR, "oa_mappings.csv")
    if os.path.exists(save_path) and not overwrite:
        return pd.read_csv(save_path, dtype=str)

    # 2011
    # https://geoportal.statistics.gov.uk/datasets/ons::
    #      output-area-to-lower-layer-super-output-area-to-middle-layer-super-output
    #      -area-to-local-authority-district-december-2011-lookup-in-england-and-wales
    #      /about
    url = (
        "https://opendata.arcgis.com/api/v3/datasets/6ecda95a83304543bc8feedbd1a58303_0"
        "/downloads/data?format=csv&spatialRefId=4326"
    )
    df2011 = pd.read_csv(url)
    df2011.drop("ObjectId", axis=1, inplace=True)

    # 2020
    # https://geoportal.statistics.gov.uk/datasets/ons::
    #         output-area-to-lower-layer-super-output-area-to-middle-layer-super-output
    #         -area-to-local-authority-district-december-2020-lookup-in-england-and
    #         -wales/about
    url = (
        "https://opendata.arcgis.com/api/v3/datasets/65664b00231444edb3f6f83c9d40591f_0"
        "/downloads/data?format=csv&spatialRefId=4326"
    )
    df2020 = pd.read_csv(url)
    df2020.drop("FID", axis=1, inplace=True)

    merged = pd.merge(df2011, df2020, how="outer")
    merged = columns_to_lowercase(merged)
    merged.to_csv(save_path, index=False)
    return merged

def download_centroids(overwrite=False):
    save_path = Path(RAW_DIR, "centroids.csv")
    if os.path.exists(save_path) and not overwrite:
        return pd.read_csv(save_path)

    # From https://geoportal.statistics.gov.uk/datasets/ons::
    #              output-areas-december-2011-population-weighted-centroids-1/about
    url = (
        "https://opendata.arcgis.com/api/v3/datasets/b0c86eaafc5a4f339eb36785628da904_0"
        "/downloads/data?format=csv&spatialRefId=27700"
    )
    df = pd.read_csv(url)
    df = columns_to_lowercase(df)
    df = df[["oa11cd", "x", "y"]]
    df.to_csv(save_path, index=False)

    return df

def columns_to_lowercase(df):
    """Convert all columns with string names in a dataframe to lowercase.

    Arguments:
        df {pd.DataFrame} -- pandas dataframe

    Returns:
        pd.DataFrame -- input dataframe with columns converted to lowercase
    """

    cols_to_rename = df.columns[[type(col) is str for col in df.columns]]
    cols_replace_dict = {name: name.lower() for name in cols_to_rename}

    return df.rename(columns=cols_replace_dict)

def query_ons_records(
    base_query, time_between_queries=1, save_path=None, overwrite=False
):
    if save_path and os.path.exists(save_path) and not overwrite:
        return gpd.read_file(save_path)

    offset_param = "&resultOffset={}"
    count_param = "&returnCountOnly=true"

    r = requests.get(base_query + count_param)
    j = r.json()
    n_records_to_query = j["count"]

    if n_records_to_query > 0:
        print("This query returns", n_records_to_query, "records.")
    else:
        raise ValueError("Input query returns no records.")

    n_queried_records = 0
    all_records = None
    while n_queried_records < n_records_to_query:
        print("PROGRESS:", n_queried_records, "out of", n_records_to_query, "records")
        start_time = time.time()

        print("Querying... ", end="")

        try:
            r = requests.get(base_query + offset_param.format(n_queried_records))

        except requests.exceptions.Timeout as e:
            print("timeout, retrying...")
            for i in range(10):
                print("attempt", i + 1)
                try:
                    r = requests.get(
                        base_query + offset_param.format(n_queried_records)
                    )
                    break
                except requests.exceptions.Timeout:
                    r = None
                    continue
            if not r:
                raise requests.exceptions.Timeout("FAILED - timeout.") from e

        j = r.json()

        n_new_records = len(j["features"])
        n_queried_records += n_new_records
        print("Got", n_new_records, "records.")

        if n_new_records > 0:
            b = bytes(r.content)
            with fiona.BytesCollection(b) as f:
                crs = f.crs
                new_records = gpd.GeoDataFrame.from_features(f, crs=crs)

            if all_records is None:
                all_records = new_records.copy(deep=True)
            else:
                all_records = all_records.append(new_records)

        if "exceededTransferLimit" in j.keys() and j["exceededTransferLimit"] is True:
            end_time = time.time()
            if end_time - start_time < time_between_queries:
                time.sleep(time_between_queries + start_time - end_time)
                continue
        else:
            print("No more records to query.")
            break

    if save_path:
        os.makedirs(save_path.parent, exist_ok=True)
        print(all_records.columns)
        all_records.to_file(save_path)

    return all_records

def lad11cd_to_lad20cd(lad11cd, mappings=None):
    if mappings is None:
        mappings = download_oa_mappings()
    return mappings[mappings.lad11cd == lad11cd]["lad20cd"].unique()

def get_oa_stats(lad20cd="E08000021"):
    """Get output area population (for each age) and place of work statistics.

    Returns:
        dict -- Dictionary of dataframe with keys population_ages and workplace.
    """
    pop_path = Path(PROCESSED_DIR, lad20cd, "population_ages.csv")
    work_path = Path(PROCESSED_DIR, lad20cd, "workplace.csv")
    if not pop_path.exists() or not work_path.exists():
        extract_la_data(lad20cd)

    population_ages = pd.read_csv(pop_path, index_col="oa11cd")
    population_ages.columns = population_ages.columns.astype(int)

    workplace = pd.read_csv(work_path, index_col="oa11cd")
    workplace = workplace["workers"]

    return {"population_ages": population_ages, "workplace": workplace}


def calc_oa_weights(
    lad20cd="E08000021",
    population_weight=1,
    workplace_weight=0,
    pop_age_groups={
        "pop_total": {"min": 0, "max": 90, "weight": 1},
        "pop_children": {"min": 0, "max": 16, "weight": 0},
        "pop_elderly": {"min": 70, "max": 90, "weight": 0},
    },
    combine=True,
):
    """Calculate weighting factor for each OA.

    Keyword Arguments:
        lad20cd {str} -- 2020 local authority district code to get output areas for (
        default E08000021, which is Newcastle upon Tyne)

        population_weight {float} -- Weighting for residential population
        (default: {1})

        workplace_weight {float} -- Weighting for workplace population
        (default: {0})

        pop_age_groups {dict} -- Residential population age groups to create
        objectives for and their corresponding weights. Dict with objective
        name as key. Each entry should be another dict with keys min (min age
        in population group), max (max age in group), and weight (objective
        weight for this group).

        combine {bool} -- If True combine all the objectives weights into a
        single overall weight using the defined weighting factors. If False
        treat all objectives separately, in which case all weights defined in
        other parameters are ignored.

    Returns:
        pd.DataFrame or pd.Series -- Weight for each OA (indexed by oa11cd) for
        each objective. Series if only one objective defined or combine is True.
    """

    data = get_oa_stats(lad20cd=lad20cd)
    population_ages = data["population_ages"]
    workplace = data["workplace"]

    if len(population_ages) != len(workplace):
        raise ValueError(
            "Lengths of inputs don't match: population_ages={}, workplace={}".format(
                len(population_ages), len(workplace)
            )
        )

    # weightings for residential population by age group
    if population_weight > 0:
        oa_population_group_weights = {}
        for name, group in pop_age_groups.items():
            # skip calculation for zeroed objectives
            if group["weight"] == 0:
                continue

            # get sum of population in group age range
            group_population = population_ages.loc[
                :,
                (population_ages.columns >= group["min"])
                & (population_ages.columns <= group["max"]),
            ].sum(axis=1)

            # normalise total population
            group_population = group_population / group_population.sum()

            # if objectives will be combined, scale by group weight
            if combine:
                group_population = group_population * group["weight"]

            oa_population_group_weights[name] = group_population

        if len(oa_population_group_weights) > 0:
            use_population = True  # some population groups with non-zero weights

            oa_population_group_weights = pd.DataFrame(oa_population_group_weights)
            if combine:
                oa_population_group_weights = oa_population_group_weights.sum(axis=1)
                oa_population_group_weights = population_weight * (
                    oa_population_group_weights / oa_population_group_weights.sum()
                )
        else:
            use_population = False  #  all population groups had zero weight
    else:
        use_population = False

    # weightings for number of workers in OA (normalised to sum to 1)
    if workplace_weight > 0:
        use_workplace = True
        workplace = workplace / workplace.sum()
        if combine:
            workplace = workplace_weight * workplace
        workplace.name = "workplace"
    else:
        use_workplace = False

    if not use_population and not use_workplace:
        raise ValueError("Must specify at least one non-zero weight.")

    if combine:
        if use_workplace and use_population:
            oa_all_weights = pd.DataFrame(
                {"workplace": workplace, "population": oa_population_group_weights}
            )
            oa_all_weights = oa_all_weights.sum(axis=1)
            return oa_all_weights / oa_all_weights.sum()
        elif use_workplace:
            return workplace
        elif use_population:
            return oa_population_group_weights
    else:
        if use_workplace and use_population:
            return oa_population_group_weights.join(workplace)
        elif use_workplace:
            return workplace
        elif use_population and len(oa_population_group_weights.columns) > 1:
            return oa_population_group_weights
        else:
            return oa_population_group_weights[oa_population_group_weights.columns[0]]
