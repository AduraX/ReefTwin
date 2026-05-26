# Data Dictionary

## IoT readings

| Field | Type | Description |
|---|---|---|
| reef_id | string | Unique reef identifier |
| timestamp | datetime | Event timestamp |
| water_temperature_c | float | Water temperature in Celsius |
| ph | float | Water pH |
| salinity_psu | float | Practical salinity units |
| turbidity_ntu | float | Turbidity in NTU |
| dissolved_oxygen_mg_l | float | Dissolved oxygen |

## NOAA-style heat-stress features

| Field | Type | Description |
|---|---|---|
| sst_celsius | float | Sea surface temperature |
| sst_anomaly_c | float | SST anomaly |
| hotspot_c | float | Heat stress hotspot |
| degree_heating_weeks | float | Cumulative thermal stress |
| bleaching_alert_area | string | Bleaching alert category |
