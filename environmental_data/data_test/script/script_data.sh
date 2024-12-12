# Shasta county area -122.532578,40.470980,-122.180328,40.720201 Lng / Lat  http://bboxfinder.com
#bbox = left,bottom,right,top
#bbox = min Longitude , min Latitude , max Longitude , max Latitud

# PARAMETERS
# NEW PARAM CASE 1:  NORTH CAL AND OREGON -125.595703,39.164141,-119.113770,44.559163
# NEW PARAM CASE 2:  TEXAS -99.063721,28.921631,-94.405518,33.596319
lon_min=-123.948
lon_max=-119.246
lat_min=35.853
lat_max=39.724

# TIME INFO
#1628485200
#1628910000
# maybe remove add or remove -u for UTC time, unsure how airnow handles data
startStamp=$(date -d"9 February 2021 12:00 AM" +%s)
endStamp=$(date -d"13 February 2021 11:00 PM" +%s) 
beginDate=20210209
endDate=20210213

# AIR NOW DATA COLLECTION
curl -X GET "https://aqs.epa.gov/data/api/sampleData/byBox?email=javiersc@umich.edu&key=carmelosprey42&param=88101&bdate=$beginDate&edate=$endDate&minlat=$lat_min&maxlat=$lat_max&minlon=$lon_min&maxlon=$lon_max" > data_airnow_case1.json
jq -r '.Data[] | [.date_local, .time_local, .site_number, .sample_measurement] | @csv' data_airnow_case1.json > data_airnow_case1.csv
echo "airnow data complete"
# COMPLETE PURPLE AIR PIPELINE
# get sensor id in some region
curl -X GET "https://api.purpleair.com/v1/sensors?fields=name%2Clatitude%2Clongitude&location_type=0&nwlng=$lon_min&nwlat=$lat_max&selng=$lon_max&selat=$lat_min" -H "X-API-Key: 432BB512-5D83-11EE-A77F-42010A800009" > purpleair_index.json
jq -r '.data[] | [.[0]] | @csv' purpleair_index.json | shuf > purpleair_index.csv
echo "purpleair index generated, pausing..."
sleep 60
# iterate and collect sensor data from timestamps etc...
idx=1
cat purpleair_index.csv | while read line;
do
	curl -X GET "https://api.purpleair.com/v1/sensors/$line/history/csv?start_timestamp=$startStamp&end_timestamp=$endStamp&average=60&fields=pm2.5_cf_1" -H "X-API-Key: 432BB512-5D83-11EE-A77F-42010A800009" > data_purpleair_case1_$idx.csv
	echo "sensor $idx done, pausing..."
	((idx=idx+1))
	sleep 60
done
