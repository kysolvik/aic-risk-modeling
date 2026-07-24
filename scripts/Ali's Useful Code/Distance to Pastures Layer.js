var area = ee.FeatureCollection('projects/ksolvik-misc/assets/Lim_Raisg');
var mapbiomas = ee.Image('projects/mapbiomas-raisg/public/collection5/mapbiomas_raisg_panamazonia_collection5_integration_v1');

print('Bandes disponibles', mapbiomas.bandNames());

var workingScale = 100;
var neighborhoodRadius = 50000; // en mètres

var availableYears = [2018, 2019, 2020, 2021, 2022];

var extrapolatedYears = [2023, 2024, 2025]; // Years that are missing data so I used the 2022 data

function computeDistanceForYear(year) {
  var pastureBinary = mapbiomas.select('classification_' + year).eq(15); //15 is the band value for Pasture
  var pasture_100m = pastureBinary
    .reduceResolution({reducer: ee.Reducer.max(), maxPixels: 1024})
    .reproject({crs: 'EPSG:4326', scale: workingScale});
  var maxDistancePixels = neighborhoodRadius / workingScale;
  var distance_pixels = pasture_100m.fastDistanceTransform(maxDistancePixels).sqrt();


  return distance_pixels
    .multiply(workingScale)
    .rename('distance_' + year);
}


var bandsAvailable = availableYears.map(computeDistanceForYear);

var distance2022 = computeDistanceForYear(2022);
var bandsExtrapolated = extrapolatedYears.map(function(year) {
  return distance2022.rename('distance_' + year);
});

var allBands = bandsAvailable.concat(bandsExtrapolated);
var stacked = ee.Image.cat(allBands).clip(area);

print('Bandes finales', stacked.bandNames());

Export.image.toAsset({
  image: stacked,
  description: 'Export_Distance_Pasture_Amazonia_100m',
  assetId: 'users/ton_nom_utilisateur/Distance_to_Pasture_100m',
  region: area.geometry().bounds(),
  crs: 'EPSG:4326',
  scale: workingScale,
  maxPixels: 1e13
});
