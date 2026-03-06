# Dataset

This project uses the **UK Road Safety Data 2022 (STATS19)**, published openly by the Department for Transport.

## Download Instructions

1. Go to: https://www.data.gov.uk/dataset/road-accidents-safety-data
2. Download these 3 files for year **2022**:
   - `Collisions_Dataset.csv`
   - `Vehicles_Dataset.csv`
   - `Casualties_Dataset.csv`
3. Place all 3 files in this `/data` folder
4. Run `notebooks/stage_1.ipynb` to begin preprocessing

## Dataset Details

| File | Description | Rows (approx) |
|---|---|---|
| `Collisions_Dataset.csv` | Accident location, severity, road/weather conditions | 106,000 |
| `Vehicles_Dataset.csv` | Vehicle type, driver demographics, manoeuvre data | 193,000 |
| `Casualties_Dataset.csv` | Casualty demographics, severity, pedestrian data | 135,000 |

## Licence

This data is published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
