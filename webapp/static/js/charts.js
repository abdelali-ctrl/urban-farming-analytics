// Charts JavaScript - Version corrigée
let chartInstances = {};

function loadYieldDistribution() {
    console.log("Loading yield distribution...");
    
    $.ajax({
        url: '/api/charts/yield-distribution',
        type: 'GET',
        success: function(response) {
            console.log("Yield distribution response:", response);
            
            if (response.success && response.data && response.layout) {
                Plotly.newPlot('yieldDistributionChart', response.data, response.layout);
                console.log("✓ Yield distribution chart loaded");
            } else {
                console.error("Failed to load yield distribution:", response.error);
                showChartError('yieldDistributionChart', 'Failed to load yield distribution');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error loading yield distribution:", error);
            showChartError('yieldDistributionChart', 'Error loading chart: ' + error);
        }
    });
}

function loadCropPerformance() {
    console.log("Loading crop performance...");
    
    $.ajax({
        url: '/api/charts/crop-performance',
        type: 'GET',
        success: function(response) {
            console.log("Crop performance response:", response);
            
            if (response.success && response.data && response.layout) {
                Plotly.newPlot('cropPerformanceChart', response.data, response.layout);
                console.log("✓ Crop performance chart loaded");
            } else {
                console.error("Failed to load crop performance:", response.error);
                showChartError('cropPerformanceChart', 'Failed to load crop performance');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error loading crop performance:", error);
            showChartError('cropPerformanceChart', 'Error loading chart: ' + error);
        }
    });
}

function loadRegionAnalysis() {
    console.log("Loading region analysis...");
    
    $.ajax({
        url: '/api/charts/region-analysis',
        type: 'GET',
        success: function(response) {
            console.log("Region analysis response:", response);
            
            if (response.success && response.data && response.layout) {
                Plotly.newPlot('regionAnalysisChart', response.data, response.layout);
                console.log("✓ Region analysis chart loaded");
            } else {
                console.error("Failed to load region analysis:", response.error);
                showChartError('regionAnalysisChart', 'Failed to load region analysis');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error loading region analysis:", error);
            showChartError('regionAnalysisChart', 'Error loading chart: ' + error);
        }
    });
}

function loadDiseaseImpact() {
    console.log("Loading disease impact...");
    
    // Get current filters (if any)
    const currentFilters = window.currentFilters || {};
    
    // Ensure we're sending valid JSON
    const requestData = Object.keys(currentFilters).length > 0 ? currentFilters : {};
    console.log("Disease impact request data:", requestData);
    
    $.ajax({
        url: '/api/data/filter',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(requestData),
        success: function(response) {
            if (response.success && response.data && response.data.length > 0) {
                createDiseaseImpactChart(response.data);
                console.log("✓ Disease impact chart created");
            } else {
                console.warn("No data for disease impact chart");
                showChartError('diseaseImpactChart', 'No data available for disease analysis');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error loading disease impact data:", error);
            showChartError('diseaseImpactChart', 'Error loading data: ' + error);
        }
    });
}

function loadClimateImpact() {
    console.log("Loading climate impact...");
    
    // Get current filters (if any)
    const currentFilters = window.currentFilters || {};
    
    $.ajax({
        url: '/api/data/filter',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(currentFilters),
        success: function(response) {
            if (response.success && response.data && response.data.length > 0) {
                createClimateImpactChart(response.data);
                console.log("✓ Climate impact chart created");
            } else {
                console.warn("No data for climate impact chart");
                showChartError('climateImpactChart', 'No data available for climate analysis');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error loading climate impact data:", error);
            showChartError('climateImpactChart', 'Error loading data: ' + error);
        }
    });
}

function createDiseaseImpactChart(data) {
    try {
        console.log("Creating disease impact chart with", data.length, "records");
        
        // Check if we have data
        if (!data || data.length === 0) {
            showChartError('diseaseImpactChart', 'No data available for disease analysis');
            return;
        }
        
        // Check if disease column exists
        const firstRecord = data[0];
        if (!firstRecord.hasOwnProperty('crop_disease_status') && !firstRecord.hasOwnProperty('crop_disease')) {
            console.warn("Disease column not found in data:", Object.keys(firstRecord));
            showChartError('diseaseImpactChart', 'Disease data not available in filtered results');
            return;
        }
        
        // Process data for disease impact
        let diseaseData = {};
        let validRecords = 0;
        
        data.forEach(item => {
            // Try different possible column names for disease
            let disease = item.crop_disease_status || item.crop_disease || 'Unknown';
            let yieldValue = item.yield_kg_per_hectare || item.yield || item.yield_kg || 0;
            
            // Only process if we have valid data
            if (disease && yieldValue > 0) {
                if (!diseaseData[disease]) {
                    diseaseData[disease] = {
                        sum: 0,
                        count: 0,
                        yields: []
                    };
                }
                
                diseaseData[disease].sum += yieldValue;
                diseaseData[disease].count++;
                diseaseData[disease].yields.push(yieldValue);
                validRecords++;
            }
        });
        
        if (validRecords === 0) {
            showChartError('diseaseImpactChart', 'No valid disease/yield data found');
            return;
        }
        
        // Calculate averages
        let diseases = Object.keys(diseaseData);
        let averages = diseases.map(d => {
            if (diseaseData[d].count > 0) {
                return diseaseData[d].sum / diseaseData[d].count;
            }
            return 0;
        });
        let counts = diseases.map(d => diseaseData[d].count);
        
        // Order by disease severity
        const diseaseOrder = ['No Disease', 'No disease', 'Healthy', 'Mild', 'Moderate', 'Severe', 'Unknown'];
        diseases = diseases.sort((a, b) => {
            const indexA = diseaseOrder.indexOf(a);
            const indexB = diseaseOrder.indexOf(b);
            return (indexA === -1 ? 999 : indexA) - (indexB === -1 ? 999 : indexB);
        });
        
        // Reorder averages and counts
        averages = diseases.map(d => diseaseData[d] ? diseaseData[d].sum / diseaseData[d].count : 0);
        counts = diseases.map(d => diseaseData[d] ? diseaseData[d].count : 0);
        
        // Create chart data
        let trace1 = {
            x: diseases,
            y: averages,
            type: 'bar',
            name: 'Average Yield',
            marker: {
                color: averages.map((avg, i) => {
                    if (avg >= 4000) return '#2ecc71';
                    if (avg >= 3000) return '#f39c12';
                    return '#e74c3c';
                })
            }
        };
        
        let trace2 = {
            x: diseases,
            y: counts,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Number of Farms',
            yaxis: 'y2',
            line: {
                color: '#3498db',
                width: 2
            },
            marker: {
                size: 8,
                color: '#3498db'
            }
        };
        
        let layout = {
            title: 'Disease Impact on Yield',
            xaxis: {
                title: 'Disease Status',
                tickangle: -45
            },
            yaxis: {
                title: 'Average Yield (kg/ha)',
                gridcolor: '#f0f0f0'
            },
            yaxis2: {
                title: 'Number of Farms',
                overlaying: 'y',
                side: 'right',
                gridcolor: '#f0f0f0'
            },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {
                color: '#333'
            },
            showlegend: true,
            legend: {
                x: 1.05,
                y: 1
            }
        };
        
        Plotly.newPlot('diseaseImpactChart', [trace1, trace2], layout);
        
    } catch (error) {
        console.error("Error creating disease impact chart:", error);
        showChartError('diseaseImpactChart', 'Error creating chart: ' + error.message);
    }
}

function createClimateImpactChart(data) {
    try {
        console.log("Creating climate impact chart with", data.length, "records");
        
        // Check if we have data
        if (!data || data.length === 0) {
            showChartError('climateImpactChart', 'No data available for climate analysis');
            return;
        }
        
        // Check if required columns exist
        const firstRecord = data[0];
        const hasTemperature = firstRecord.hasOwnProperty('temperature_C') || firstRecord.hasOwnProperty('temperature');
        const hasMoisture = firstRecord.hasOwnProperty('soil_moisture_') || firstRecord.hasOwnProperty('soil_moisture');
        
        if (!hasTemperature && !hasMoisture) {
            console.warn("Climate columns not found:", Object.keys(firstRecord));
            showChartError('climateImpactChart', 'Climate data not available in filtered results');
            return;
        }
        
        // Process data for climate impact
        let tempRanges = ['<20°C', '20-25°C', '25-30°C', '>30°C'];
        let moistureRanges = ['<30%', '30-50%', '50-70%', '>70%'];
        
        let tempData = {};
        let moistureData = {};
        let validRecords = 0;
        
        data.forEach(item => {
            // Try different column names
            let temp = item.temperature_C || item.temperature || 0;
            let moisture = item.soil_moisture_ || item.soil_moisture || 0;
            let yieldValue = item.yield_kg_per_hectare || item.yield || item.yield_kg || 0;
            
            // Only process if we have some valid data
            if ((temp > 0 || moisture > 0) && yieldValue > 0) {
                // Categorize temperature
                if (temp > 0) {
                    let tempRange;
                    if (temp < 20) tempRange = '<20°C';
                    else if (temp <= 25) tempRange = '20-25°C';
                    else if (temp <= 30) tempRange = '25-30°C';
                    else tempRange = '>30°C';
                    
                    if (!tempData[tempRange]) {
                        tempData[tempRange] = { sum: 0, count: 0 };
                    }
                    tempData[tempRange].sum += yieldValue;
                    tempData[tempRange].count++;
                }
                
                // Categorize moisture
                if (moisture > 0) {
                    let moistureRange;
                    if (moisture < 30) moistureRange = '<30%';
                    else if (moisture <= 50) moistureRange = '30-50%';
                    else if (moisture <= 70) moistureRange = '50-70%';
                    else moistureRange = '>70%';
                    
                    if (!moistureData[moistureRange]) {
                        moistureData[moistureRange] = { sum: 0, count: 0 };
                    }
                    moistureData[moistureRange].sum += yieldValue;
                    moistureData[moistureRange].count++;
                }
                
                validRecords++;
            }
        });
        
        if (validRecords === 0) {
            showChartError('climateImpactChart', 'No valid climate/yield data found');
            return;
        }
        
        // Calculate averages
        let tempAverages = tempRanges.map(range => {
            if (tempData[range] && tempData[range].count > 0) {
                return tempData[range].sum / tempData[range].count;
            }
            return null;
        });
        
        let moistureAverages = moistureRanges.map(range => {
            if (moistureData[range] && moistureData[range].count > 0) {
                return moistureData[range].sum / moistureData[range].count;
            }
            return null;
        });
        
        // Filter out null values
        const hasTempData = tempAverages.some(v => v !== null);
        const hasMoistureData = moistureAverages.some(v => v !== null);
        
        if (!hasTempData && !hasMoistureData) {
            showChartError('climateImpactChart', 'Insufficient climate data for analysis');
            return;
        }
        
        // Create chart based on available data
        let traces = [];
        let layout;
        
        if (hasTempData && hasMoistureData) {
            // Create subplot with both
            traces.push({
                x: tempRanges,
                y: tempAverages,
                type: 'bar',
                name: 'By Temperature',
                marker: { color: '#e74c3c' }
            });
            
            traces.push({
                x: moistureRanges,
                y: moistureAverages,
                type: 'bar',
                name: 'By Soil Moisture',
                marker: { color: '#3498db' }
            });
            
            layout = {
                grid: { 
                    rows: 1, 
                    columns: 2, 
                    pattern: 'independent',
                    xgap: 0.1
                },
                title: 'Climate Impact on Yield',
                xaxis: { title: 'Temperature Ranges', tickangle: -45 },
                xaxis2: { title: 'Soil Moisture Ranges', tickangle: -45 },
                yaxis: { title: 'Average Yield (kg/ha)' },
                yaxis2: { title: 'Average Yield (kg/ha)' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#333' },
                showlegend: true,
                legend: { x: 1.05, y: 1 }
            };
        } else if (hasTempData) {
            // Only temperature data
            traces.push({
                x: tempRanges,
                y: tempAverages,
                type: 'bar',
                name: 'Temperature Impact',
                marker: { color: '#e74c3c' }
            });
            
            layout = {
                title: 'Temperature Impact on Yield',
                xaxis: { title: 'Temperature Ranges', tickangle: -45 },
                yaxis: { title: 'Average Yield (kg/ha)' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#333' }
            };
        } else {
            // Only moisture data
            traces.push({
                x: moistureRanges,
                y: moistureAverages,
                type: 'bar',
                name: 'Soil Moisture Impact',
                marker: { color: '#3498db' }
            });
            
            layout = {
                title: 'Soil Moisture Impact on Yield',
                xaxis: { title: 'Soil Moisture Ranges', tickangle: -45 },
                yaxis: { title: 'Average Yield (kg/ha)' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#333' }
            };
        }
        
        Plotly.newPlot('climateImpactChart', traces, layout);
        
    } catch (error) {
        console.error("Error creating climate impact chart:", error);
        showChartError('climateImpactChart', 'Error creating chart: ' + error.message);
    }
}

function showChartError(chartId, message) {
    const container = document.getElementById(chartId);
    if (container) {
        container.innerHTML = `
            <div class="chart-error-placeholder text-center py-5">
                <div class="mb-3">
                    <i class="fas fa-chart-line fa-3x text-muted"></i>
                </div>
                <h6 class="text-muted mb-2">${message}</h6>
                <p class="small text-muted mb-0">
                    <i class="fas fa-lightbulb me-1"></i>
                    Try adjusting your filters or click "Reset"
                </p>
            </div>
        `;
    }
}

// Update charts when window resizes
$(window).resize(function() {
    console.log("Window resized, updating charts...");
    
    // Update all Plotly charts
    const chartIds = [
        'yieldDistributionChart',
        'cropPerformanceChart', 
        'regionAnalysisChart',
        'diseaseImpactChart',
        'climateImpactChart'
    ];
    
    chartIds.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart && chart.data) {
            Plotly.Plots.resize(chartId);
        }
    });
});

// Export functions for use in dashboard.js
window.chartFunctions = {
    loadYieldDistribution,
    loadCropPerformance,
    loadRegionAnalysis,
    loadDiseaseImpact,
    loadClimateImpact
};

// charts.js - Add this at the very end

// Make sure functions are available globally
console.log("charts.js loaded successfully");

// Export all functions
if (typeof window !== 'undefined') {
    window.chartFunctions = {
        loadYieldDistribution,
        loadCropPerformance,
        loadRegionAnalysis,
        loadDiseaseImpact,
        loadClimateImpact,
        createDiseaseImpactChart,
        createClimateImpactChart
    };
}