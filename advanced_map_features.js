// Advanced Map Features for Punjab Soil Predictor
// Add this to your index.html script section for additional functionality

// ============================================================================
// FEATURE 1: Search by City/Location Name
// ============================================================================
function addLocationSearch() {
    // Punjab cities with coordinates
    const punjabCities = {
        'Ludhiana': { lat: 30.9010, lng: 75.8573 },
        'Amritsar': { lat: 31.6340, lng: 74.8723 },
        'Jalandhar': { lat: 31.3260, lng: 75.5762 },
        'Patiala': { lat: 30.3398, lng: 76.3869 },
        'Bathinda': { lat: 30.2110, lng: 74.9455 },
        'Hoshiarpur': { lat: 31.5334, lng: 75.9120 },
        'Mohali': { lat: 30.7046, lng: 76.7179 },
        'Pathankot': { lat: 32.2746, lng: 75.6521 },
        'Moga': { lat: 30.8162, lng: 75.1714 },
        'Kapurthala': { lat: 31.3800, lng: 75.3800 },
        'Chandigarh': { lat: 30.7333, lng: 76.7794 },
        'Firozpur': { lat: 30.9257, lng: 74.6142 },
        'Gurdaspur': { lat: 32.0407, lng: 75.4053 },
        'Faridkot': { lat: 30.6705, lng: 74.7600 },
        'Sangrur': { lat: 30.2453, lng: 75.8430 }
    };

    // Create search control HTML
    const searchHTML = `
        <div style="position: absolute; top: 10px; left: 60px; z-index: 1000; background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
            <input type="text" id="citySearch" list="cities" placeholder="Search city..." 
                   style="padding: 8px; border: 2px solid #667eea; border-radius: 6px; width: 200px;">
            <datalist id="cities">
                ${Object.keys(punjabCities).map(city => `<option value="${city}">`).join('')}
            </datalist>
            <button onclick="searchCity()" style="padding: 8px 15px; background: #667eea; color: white; border: none; border-radius: 6px; margin-left: 5px; cursor: pointer;">Go</button>
        </div>
    `;

    document.getElementById('map').insertAdjacentHTML('afterbegin', searchHTML);

    window.searchCity = function() {
        const cityName = document.getElementById('citySearch').value;
        const cityCoords = punjabCities[cityName];
        
        if (cityCoords) {
            map.setView([cityCoords.lat, cityCoords.lng], 12);
            setMarker(cityCoords.lat, cityCoords.lng);
            updateCoordinates(cityCoords.lat, cityCoords.lng);
        } else {
            alert('City not found. Please select from the dropdown.');
        }
    };
}

// ============================================================================
// FEATURE 2: Prediction History on Map
// ============================================================================
class PredictionHistory {
    constructor() {
        this.predictions = [];
        this.markers = [];
        this.loadFromStorage();
    }

    add(lat, lng, depth, predictions) {
        const prediction = {
            id: Date.now(),
            lat: lat,
            lng: lng,
            depth: depth,
            timestamp: new Date().toISOString(),
            predictions: predictions
        };

        this.predictions.unshift(prediction);
        
        // Keep only last 20 predictions
        if (this.predictions.length > 20) {
            this.predictions = this.predictions.slice(0, 20);
        }

        this.saveToStorage();
        this.addMarkerToMap(prediction);
    }

    addMarkerToMap(prediction) {
        if (!map) return;

        const smallIcon = L.divIcon({
            className: 'history-marker',
            html: '<div style="background: rgba(102, 126, 234, 0.3); width: 15px; height: 15px; border-radius: 50%; border: 2px solid #667eea;"></div>',
            iconSize: [15, 15],
            iconAnchor: [7, 7]
        });

        const historyMarker = L.marker([prediction.lat, prediction.lng], { icon: smallIcon });
        
        // Create popup with prediction summary
        const nValue = prediction.predictions['N-value'];
        const popupContent = `
            <div style="min-width: 200px;">
                <strong>📊 Previous Prediction</strong><br>
                <small>${new Date(prediction.timestamp).toLocaleString()}</small><br>
                <hr style="margin: 5px 0;">
                Depth: ${prediction.depth}m<br>
                ${nValue ? `N-value: ${nValue.value.toFixed(2)}` : ''}
            </div>
        `;

        historyMarker.bindPopup(popupContent);
        historyMarker.addTo(map);
        
        this.markers.push(historyMarker);
    }

    clearMarkers() {
        this.markers.forEach(marker => map.removeLayer(marker));
        this.markers = [];
    }

    showAll() {
        this.clearMarkers();
        this.predictions.forEach(pred => this.addMarkerToMap(pred));
    }

    saveToStorage() {
        try {
            localStorage.setItem('predictionHistory', JSON.stringify(this.predictions));
        } catch (e) {
            console.log('Could not save to localStorage');
        }
    }

    loadFromStorage() {
        try {
            const stored = localStorage.getItem('predictionHistory');
            if (stored) {
                this.predictions = JSON.parse(stored);
            }
        } catch (e) {
            console.log('Could not load from localStorage');
        }
    }

    clear() {
        this.predictions = [];
        this.clearMarkers();
        this.saveToStorage();
    }
}

const predictionHistory = new PredictionHistory();

// ============================================================================
// FEATURE 3: Layer Controls (Satellite, Terrain, etc.)
// ============================================================================
function addLayerControls() {
    // Base layers
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    });

    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: '© Esri'
    });

    const terrainLayer = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenTopoMap contributors'
    });

    const baseMaps = {
        "Street Map": osmLayer,
        "Satellite": satelliteLayer,
        "Terrain": terrainLayer
    };

    // Add layer control
    L.control.layers(baseMaps).addTo(map);

    // Set default
    osmLayer.addTo(map);
}

// ============================================================================
// FEATURE 4: Drawing Tools (Measure Distance, Draw Areas)
// ============================================================================
function addDrawingTools() {
    // Add measuring tool
    let measureLine;
    let measurePoints = [];
    let isMeasuring = false;

    const measureControl = L.control({ position: 'topleft' });
    
    measureControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'leaflet-bar');
        div.innerHTML = `
            <a href="#" title="Measure distance" style="font-size: 20px; line-height: 30px; width: 30px; height: 30px; display: block; text-align: center;">
                📏
            </a>
        `;
        
        div.onclick = function(e) {
            e.preventDefault();
            toggleMeasure();
        };
        
        return div;
    };

    measureControl.addTo(map);

    function toggleMeasure() {
        isMeasuring = !isMeasuring;
        if (!isMeasuring) {
            clearMeasure();
        } else {
            alert('Click on map to measure distance. Click again to see result.');
        }
    }

    function clearMeasure() {
        if (measureLine) {
            map.removeLayer(measureLine);
        }
        measurePoints = [];
    }

    map.on('click', function(e) {
        if (isMeasuring) {
            measurePoints.push(e.latlng);
            
            if (measurePoints.length === 2) {
                const distance = measurePoints[0].distanceTo(measurePoints[1]);
                
                measureLine = L.polyline(measurePoints, {
                    color: '#667eea',
                    weight: 3,
                    dashArray: '10, 5'
                }).addTo(map);

                const midpoint = [
                    (measurePoints[0].lat + measurePoints[1].lat) / 2,
                    (measurePoints[0].lng + measurePoints[1].lng) / 2
                ];

                L.popup()
                    .setLatLng(midpoint)
                    .setContent(`Distance: ${(distance / 1000).toFixed(2)} km`)
                    .openOn(map);

                isMeasuring = false;
                measurePoints = [];
            }
        }
    });
}

// ============================================================================
// FEATURE 5: Geolocation - Find My Location
// ============================================================================
function addGeolocationControl() {
    const geoControl = L.control({ position: 'topleft' });
    
    geoControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'leaflet-bar');
        div.innerHTML = `
            <a href="#" title="Find my location" style="font-size: 20px; line-height: 30px; width: 30px; height: 30px; display: block; text-align: center;">
                📍
            </a>
        `;
        
        div.onclick = function(e) {
            e.preventDefault();
            findMyLocation();
        };
        
        return div;
    };

    geoControl.addTo(map);

    function findMyLocation() {
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser');
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;

                // Check if within Punjab bounds
                if (lat >= 30.0 && lat <= 32.0 && lng >= 74.0 && lng <= 77.0) {
                    map.setView([lat, lng], 13);
                    setMarker(lat, lng);
                    updateCoordinates(lat, lng);
                    
                    L.circle([lat, lng], {
                        radius: position.coords.accuracy,
                        color: '#667eea',
                        fillColor: '#667eea',
                        fillOpacity: 0.1
                    }).addTo(map);
                } else {
                    alert('Your location is outside Punjab region. Showing Punjab center instead.');
                    map.setView([31.1471, 75.3412], 8);
                }
            },
            (error) => {
                alert('Unable to retrieve your location: ' + error.message);
            }
        );
    }
}

// ============================================================================
// FEATURE 6: Fullscreen Map
// ============================================================================
function addFullscreenControl() {
    const fullscreenControl = L.control({ position: 'topright' });
    
    fullscreenControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'leaflet-bar');
        div.innerHTML = `
            <a href="#" id="fullscreenBtn" title="Fullscreen" style="font-size: 20px; line-height: 30px; width: 30px; height: 30px; display: block; text-align: center;">
                ⛶
            </a>
        `;
        
        div.onclick = function(e) {
            e.preventDefault();
            toggleFullscreen();
        };
        
        return div;
    };

    fullscreenControl.addTo(map);

    function toggleFullscreen() {
        const mapContainer = document.getElementById('map');
        
        if (!document.fullscreenElement) {
            mapContainer.requestFullscreen().then(() => {
                setTimeout(() => map.invalidateSize(), 100);
            });
        } else {
            document.exitFullscreen().then(() => {
                setTimeout(() => map.invalidateSize(), 100);
            });
        }
    }
}

// ============================================================================
// Initialize all features (call this after map is created)
// ============================================================================
function initializeAdvancedMapFeatures() {
    if (!map) {
        console.error('Map not initialized');
        return;
    }

    addLocationSearch();
    addLayerControls();
    addDrawingTools();
    addGeolocationControl();
    addFullscreenControl();
    
    // Show prediction history on map
    predictionHistory.showAll();

    console.log('✓ Advanced map features initialized');
}

// ============================================================================
// Update prediction function to save to history
// ============================================================================
// Modify your existing predict() function to include:
// After successful prediction:
/*
if (data.success) {
    predictionHistory.add(
        data.input.latitude,
        data.input.longitude,
        data.input.depth,
        data.predictions
    );
}
*/