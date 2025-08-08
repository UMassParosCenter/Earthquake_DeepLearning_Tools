"""
Script: earthquake_event_waveform_exporter.py

This script extracts waveform data for a catalog of earthquake events by querying a
Paroscientific sensor's InfluxDB using estimated surface wave arrival times.

It performs the following tasks:
1. Loads a CSV earthquake catalog (`EarthQuakeData.csv`).
2. Cleans and parses the catalog into structured data (e.g., time, lat/lon, magnitude).
3. For each event:
   - Calculates expected infrasound (surface wave) arrival delay using geodesic distance.
   - Queries waveform data from InfluxDB ±15s to ±45s around the predicted arrival time.
   - Stores waveform arrays and event metadata (time, location, magnitude, etc.).
4. Exports all successful event waveforms and metadata into a single `.pkl` file
   (`EarthQuakeEvents.pkl`) for downstream processing or machine learning.

Components:
- `InfrasoundUtils`: Computes travel-time delay based on surface wave velocity.
- `EarthquakeCatalog`: Handles loading and cleaning the CSV earthquake catalog.
- `EarthquakeDataExporter`: Coordinates querying and saving waveform data.

Inputs:
- Earthquake CSV file: `EarthQuakeData.csv`
- InfluxDB connection: via `query_influx_data()` from `paros_data_grabber`

Outputs:
- Pickled dictionary of waveform data + metadata: `Exported_Paros_Data/EarthQuakeEvents.pkl`

Key Parameters:
- `station_lat/lon`: Coordinates of the sensor station
- `vsurface`: Assumed Rayleigh wave velocity (3.4 km/s)
- `time_before` / `time_after`: Time padding around arrival window
- `box_id`, `sensor_id`, `password`: Required for data access

Dependencies:
- geopy, pandas, tqdm, pickle, datetime, pathlib
- Requires access to a valid InfluxDB and `paros_data_grabber` module

Note: Events with no waveform data are skipped and a warning is logged via `tqdm.write`.

Ethan Gelfand, 08/06/2025
"""


from datetime import timedelta
from pathlib import Path
from geopy.distance import geodesic
import pandas as pd
import pickle
from paros_data_grabber import query_influx_data
from tqdm import tqdm


class InfrasoundUtils:
    @staticmethod
    def surface_wave_delay(event_lat, event_lon, station_lat, station_lon):
        dist_km = geodesic((event_lat, event_lon), (station_lat, station_lon)).km
        vsurface = 3.4  # km/s typical Rayleigh wave group velocity 
        return dist_km / vsurface


class EarthquakeCatalog:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self._clean()

    def _clean(self):
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.df['time'] = pd.to_datetime(self.df['time'], errors='coerce')
        self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
        self.df['depth'] = pd.to_numeric(self.df['depth'], errors='coerce')
        self.df['mag'] = pd.to_numeric(self.df['mag'], errors='coerce')
        self.df['magtype'] = self.df['magtype'].str.strip().str.lower()
        self.df.dropna(subset=['time'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def get_events(self):
        return self.df.iterrows()


class EarthquakeDataExporter:
    def __init__(self, station_lat, station_lon, box_id, sensor_id, password, output_path,
                 time_before=timedelta(seconds=15), time_after=timedelta(seconds=45)):
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.box_id = box_id
        self.sensor_id = sensor_id
        self.password = password
        self.time_before = time_before
        self.time_after = time_after
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_path / "EarthQuakeEvents.pkl"
        self.data_dict = {}
        self.counter = 1

    def process_event(self, idx, row):
        try:
            event_time = row['time']
            event_lat = row['latitude']
            event_lon = row['longitude']

            # Calculate fixed surface wave delay
            delay = InfrasoundUtils.surface_wave_delay(event_lat, event_lon, self.station_lat, self.station_lon)
            arrival_time = event_time + timedelta(seconds=delay)

            start_time = (arrival_time - self.time_before).strftime("%Y-%m-%dT%H:%M:%S")
            end_time = (arrival_time + self.time_after).strftime("%Y-%m-%dT%H:%M:%S")

            data = query_influx_data(
                start_time=start_time,
                end_time=end_time,
                box_id=self.box_id,
                sensor_id=self.sensor_id,
                password=self.password
            )

            if not data:
                tqdm.write(f"[Warning] No data for event {idx+1} at {event_time}")
                return

            data_arrays = {key: df_.values for key, df_ in data.items()}
            metadata = {
                'time': event_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                'latitude': event_lat,
                'longitude': event_lon,
                'depth': row['depth'],
                'magnitude': row['mag'],
                'magtype': row['magtype'],
                'arrival_time': arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            }

            key = f"event_{self.counter:03d}"
            self.data_dict[key] = {
                'waveform': data_arrays,
                'metadata': metadata
            }
            self.counter += 1

        except Exception as e:
            tqdm.write(f"[Error] Event {idx+1} failed: {e}")

    def export(self):
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.data_dict, f)
        print(f"[Done] Data saved to {self.output_file.resolve()}")


if __name__ == "__main__":
    catalog_path = "EarthQuakeData.csv"
    output_dir = "Exported_Paros_Data"

    station_lat, station_lon = 24.07396028832464, 121.1286975322632
    box_id = "parost2"
    sensor_id = "141929"
    password = "******" # Replace with actual password

    catalog = EarthquakeCatalog(catalog_path)
    exporter = EarthquakeDataExporter(
        station_lat=station_lat,
        station_lon=station_lon,
        box_id=box_id,
        sensor_id=sensor_id,
        password=password,
        output_path=output_dir
    )

    # Clean event loop with tqdm
    for idx in tqdm(range(len(catalog.df)), desc="Processing Events", colour="green"):
        row = catalog.df.iloc[idx]
        exporter.process_event(idx, row)

    exporter.export()