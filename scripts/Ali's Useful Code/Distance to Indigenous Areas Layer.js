var area = ee.FeatureCollection('projects/ksolvik-misc/assets/Lim_Raisg');
var wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons");

var workingScale = 100;
var neighborhoodRadius = 50000;
var maxDistancePixels = neighborhoodRadius / workingScale;

var indigenousAreas = wdpa.filter(ee.Filter.eq('GOV_TYPE', 'Indigenous Peoples'))
                          .filterBounds(area);

var referenceImage = ee.Image(0).byte()
  .paint(indigenousAreas, 1)
  .reproject({crs: 'EPSG:4326', scale: workingScale});

var distance_pixels = referenceImage.fastDistanceTransform(maxDistancePixels).sqrt();
var distance = distance_pixels.multiply(workingScale).rename('distance_IndigenousAreas');

Export.image.toAsset({
  image: distance,
  description: 'Export_Distance_Indigenous_Amazonia_100m',
  assetId: 'users/ton_nom_utilisateur/DistanceIndigenousAreas_Amazon_100m',
  region: area.geometry().bounds(),
  crs: 'EPSG:4326',
  scale: workingScale,
  maxPixels: 1e13
});

