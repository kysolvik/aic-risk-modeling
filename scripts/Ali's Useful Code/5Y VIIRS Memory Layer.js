var area = ee.FeatureCollection('projects/ksolvik-misc/assets/Lim_Raisg');
var viirs_collection = ee.ImageCollection('projects/macedo-lab-general-9051/assets/viirs_snpp_archive');
var targetYears = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];

var creerHistoriqueFeu = function(annee) {
  var annee_fin = ee.Number(annee).subtract(1);
  var date_debut = ee.Date.fromYMD(annee_fin.subtract(4), 1, 1);
  var date_fin = ee.Date.fromYMD(annee_fin, 11, 1);

  var feux_5_ans = viirs_collection.filterDate(date_debut, date_fin);
  var masque_binaire = feux_5_ans.max();
  var nom_bande = ee.String('target_').cat(ee.Number(annee).format('%d'));

  var couche_finale = masque_binaire.unmask(0).rename(nom_bande);

  return couche_finale;
};

var liste_images = targetYears.map(creerHistoriqueFeu);
var stacked = ee.Image.cat(liste_images).clip(area);

print('Bandes finales', stacked.bandNames());

Export.image.toAsset({
  image: stacked,
  description: 'Export_VIIRSfireMemory5y_Amazon_100m',
  assetId: 'projects/columbia-research-project/assets/VIIRSfireMemory5y_Amazon_100m',
  region: area.geometry().bounds(),
  crs: 'EPSG:4326',
  scale: 100,
  maxPixels: 1e13
});
