# Model Card: Bleaching Risk Model

## Model purpose

Predict near-term coral bleaching risk for a reef using environmental and heat-stress features.

## Inputs

- water temperature
- pH
- salinity
- turbidity
- dissolved oxygen
- SST anomaly
- HotSpot
- Degree Heating Weeks
- 7-day temperature trend

## Output

- bleaching_risk_score: 0.0 to 1.0
- risk_category: normal, watch, warning, alert

## Limitations

This starter model uses synthetic/sample data. It is not suitable for real conservation decisions until trained and validated on verified reef monitoring datasets.

## Intended use

Portfolio, engineering demonstration, and research prototype.
