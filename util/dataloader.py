import os
from bs4 import BeautifulSoup
import requests as rq
import zipfile
import pandas as pd
import pickle
import functools

class Loader:
    ZIP_NAME = "data.zip"
    DATA_BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes"
    # KINDS = ["wind", "air_temperature", "precipitation", "solar"]

    def __init__(self, metrics: list, data_folder: str, station_id: str = "02115"):
        """
        metrics is a list of "wind", "air_temperature", "precipitation" and/or "solar"
        """
        self.station_id = station_id
        self.metrics = metrics
        self.data_folder = data_folder
        self.contents_path = os.path.join(data_folder, "contents.pickle")
        self.metric_urls = { metric: f"{self.DATA_BASE_URL}/{metric}/historical/" for metric in metrics }

    def query_metric(self, metric) -> tuple: 
        """
        Queries a specific metrich (such as wind) and returns dictionaries (mapping from filename to request)
        for the meta data, the descriptions and the actual data csv files.
        """
        def seach_refs(soup, base_url, keyword) -> dict:
            relevant_links = [a.get("href") for a in soup.find_all("a", href=True) if a.get("href").__contains__(keyword)]
            resps = { name: rq.get(base_url + name) for name in relevant_links }
            return resps

        url = self.metric_urls[metric]
        desc_url = f"{self.DATA_BASE_URL}/{metric}/"
        meta_url = f"{self.DATA_BASE_URL}/{metric}/meta_data/"

        desc_soup = BeautifulSoup(rq.get(desc_url).text, "html.parser")
        descs_d = seach_refs(desc_soup, desc_url, "pdf")

        meta_soup = BeautifulSoup(rq.get(meta_url).text, "html.parser")
        meta_d = seach_refs(meta_soup, meta_url, self.station_id)

        data_soup = BeautifulSoup(rq.get(url).text, "html.parser")
        csvs_d = seach_refs(data_soup, url, self.station_id)

        return meta_d, descs_d, csvs_d
    
    def download_metric(self, metric) -> tuple:
        """
        Downloads all meta files, descriptions and csv files for an metric to
        the `data_folder`. Returns a tuple of lists with the filesnames for meta
        data, descriptions and csvs.
        """
        meta, descs, csvs = self.query_metric(metric)
        save_path = os.path.join(self.data_folder, metric)
        os.makedirs(os.path.join(save_path, "meta"), exist_ok=True)

        # save dataset description pdfs
        desc_file_paths = []
        for descr, resp in descs.items():
            desc_file_path = os.path.join(save_path, "meta", descr)
            with open(desc_file_path, "wb") as file:
                file.write(resp.content)
            desc_file_paths.append(desc_file_path)

        # save meta description
        meta_file_paths = []
        for meta_data, resp in meta.items():
            zip_path = os.path.join(save_path, "meta", meta_data)

            with open(zip_path, "wb") as z:
                z.write(resp.content)

            with zipfile.ZipFile(zip_path, "r") as zip_file:
                for filename in zip_file.namelist():
                    meta_file_paths.append(os.path.abspath(os.path.join(save_path, "meta", filename)))
                zip_file.extractall(os.path.join(save_path, "meta"))
            
            os.remove(zip_path)

        # save dataset csvs
        csv_file_paths = []
        for csv, csv_resp in csvs.items():
            zip_path = os.path.join(save_path, csv)

            # write zip to disk
            with open(zip_path, "wb") as z:
                z.write(csv_resp.content)

            # extract zip contents
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                for filename in zip_file.namelist():
                    csv_file_paths.append(os.path.abspath(os.path.join(save_path, filename)))
                zip_file.extractall(save_path)

            os.remove(zip_path)

        return desc_file_paths, meta_file_paths, csv_file_paths


    def download_all_metrics(self, reset=False): 
        """
    	Downloads all metrics specified in `self.metrics`. Returns a dictionary
    	mapping from metric to a list of csv file paths for later loading.
        """
        metric_files = {}

        # download data only the first time
        if reset or not os.path.isfile(self.contents_path):
            os.makedirs(self.data_folder, exist_ok=True)

            for metric in self.metrics:
                m_desc_path, _, m_csv_path = self.download_metric(metric)
                metric_files[metric] =  m_csv_path

            with open(os.path.join(self.contents_path), "wb") as fh:
                pickle.dump(metric_files, fh, protocol=pickle.HIGHEST_PROTOCOL)
        # only load the dictionary containing abs. file paths for the csv files for each metric
        else:
            with open(self.contents_path, "rb") as fh:
                metric_files = pickle.load(fh)

        self.metric_files = metric_files
        return metric_files

    @functools.cached_property
    def as_dataframe(self):
        """
        Save all metrics to disk and returns a pandas dataframe containing all
        joint metrics and a list of all seperate metrics. Note that this all
        this basic pre-processing, like properly parsing the date and
        classifying -999 values as NaN (as per the data description).
        """
        if len(list(self.metric_files.items())) == 0:
            self.download_all_metrics()

        metric_dfs = { kind: None for kind in self.metrics }
        for metric, files in self.metric_files.items():
            dfs = []
            for file in files:
                df = pd.read_csv(file, sep=";", na_values=-999)
                df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H%M")
                dfs.append(df)
            df = pd.concat(dfs)
            df.sort_values(by="MESS_DATUM", inplace=True)
            df.columns = map(lambda c: c if c == "STATIONS_ID" or c == "MESS_DATUM" else f"{c}_{metric}", df.columns)
            metric_dfs[metric] = df

        # create a single dataframe with all metric merged
        dfs = list(metric_dfs.values())
        if len(self.metrics) > 1:
            df = pd.merge(dfs[0], dfs[1], on=["MESS_DATUM", "STATIONS_ID"], how="inner", suffixes=tuple(list(map(lambda x: "_" + x, metric_dfs.keys()))[:2]))
            if len(self.metrics) > 2:
            # Loop through the remaining DataFrames and merge with the result
                for i, df1 in enumerate(dfs[2:]):
                    df = pd.merge(df, df1, on=["MESS_DATUM", "STATIONS_ID"], how="inner", suffixes=(None, "_" + list(metric_dfs.keys())[i+2]))
            return metric_dfs, df
        else:
            return metric_dfs, list(metric_dfs.values())[0]
            
