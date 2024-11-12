# Spotify Streaming History Analysis

This project analyzes your Spotify streaming history to provide insights into your listening habits.


## Data

The `data` directory contains JSON files with your Spotify streaming history and a CSV file `final.csv`.

## Getting Started

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required packages:**
    ```sh
    pip install pandas
    ```

3. **Open the Jupyter Notebooks:**
    ```sh
    jupyter notebook Spotify_Streaming_History_Analysis.ipynb
    jupyter notebook Spotify_Clustering_Analysis.ipynb
    ```

## Analysis

The Jupyter Notebooks contain the following sections:

### Spotify_Streaming_History_Analysis.ipynb

1. **Data Loading:**
    - Loads the JSON files from the `data` directory.
    - Converts the data into a pandas DataFrame.

2. **Analysis:**
    - Provides insights into your most listened-to artists.

### Spotify_Clustering_Analysis.ipynb

1. **Data Loading:**
    - Loads the CSV file `final.csv` from the `data` directory.
    - Converts the data into a pandas DataFrame.

2. **Clustering Analysis:**
    - Performs clustering analysis on the data.

## Usage

Run the cells in the Jupyter Notebooks to load the data and perform the analysis.