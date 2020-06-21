# Microsoft Azure Hackathon Readme
### Team: 84k PA
This is the Grab challenge API instruction manual.

## API Usage

> [POST] http://20.195.8.84/api/v1/service/grab-eta/score

### Example Input
Below is the example JSON to be posted in the API endpoint.
```json
{
    "latitude_origin": -6.141255,
    "longitude_origin": 106.692710,
    "latitude_destination": -6.141150,
    "longitude_destination": -6.141150,
    "timestamp": 1590487113,
    "hour_of_day": 9,
    "day_of_week": 1
}
```
### Example Output
Below is the expected outout from the API endpoint.
```json
{
  "Results": {
    "WebServiceInput0": [
      {
        "latitude_origin": -6.141255,
        "longitude_origin": 106.69271,
        "latitude_destination": -6.14115,
        "longitude_destination": -6.14115,
        "timestamp": 1590487113,
        "hour_of_day": 9,
        "day_of_week": 1,
        "eta": 1597
      }
    ]
  }
}
```

### Note
The weight "lightgbm.pkl" is not added due to Git's size constraint of 100MB.