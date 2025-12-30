// analysis.js - Complete analysis dashboard functionality

let analysisData = null;

function initializeAnalysis() {
    console.log("Initializing analysis dashboard...");
    
    // Load data first
    loadAnalysisData();
    
    // Setup event listeners
    setupAnalysisListeners();
}

function loadAnalysisData() {
    console.log("Loading analysis data...");
    
    $.ajax({
        url: '/api/data/analysis',  // Changed from '/api/data/all'
        type: 'GET',
        success: function(response) {
            if (response.success && response.data) {
                analysisData = response.data;
                console.log(`Loaded ${analysisData.length} records for analysis`);
                
                // Now load all charts
                loadCorrelationAnalysis();
                loadTimeSeriesAnalysis();
                loadDistributionAnalysis();
                loadComparisonAnalysis();
                loadHeatmapAnalysis();
                loadStatisticalSummary();
            } else {
                console.error("Failed to load analysis data:", response);
                showAnalysisError('Failed to load data for analysis');
            }
        },
        error: function(error) {
            console.error("Error loading analysis data:", error);
            showAnalysisError('Error loading data: ' + error);
        }
    });
}

function setupAnalysisListeners() {
    // Correlation analysis controls
    $('#correlationX, #correlationY, #correlationColor').on('change', function() {
        loadCorrelationAnalysis();
    });
    
    // Time series analysis controls
    $('#timeVariable, #timeMetric, #timeAggregation, #timeGroup').on('change', function() {
        loadTimeSeriesAnalysis();
    });
    
    // Distribution analysis controls
    $('#distributionVar, #distributionType').on('change', function() {
        loadDistributionAnalysis();
    });
    
    // Comparison analysis controls
    $('#compareBy, #compareMetric').on('change', function() {
        loadComparisonAnalysis();
    });
    
    // Heatmap analysis controls
    $('#heatmapX, #heatmapY, #heatmapMetric, #heatmapAgg').on('change', function() {
        loadHeatmapAnalysis();
    });
}

function loadCorrelationAnalysis() {
    if (!analysisData || analysisData.length === 0) {
        console.warn("No data available for correlation analysis");
        return;
    }
    
    const xVar = $('#correlationX').val();
    const yVar = $('#correlationY').val();
    const colorVar = $('#correlationColor').val();
    
    console.log(`Loading correlation: ${xVar} vs ${yVar} colored by ${colorVar}`);
    
    // Filter out invalid data
    const validData = analysisData.filter(d => 
        d[xVar] !== null && d[yVar] !== null && 
        !isNaN(d[xVar]) && !isNaN(d[yVar])
    );
    
    if (validData.length === 0) {
        showChartError('correlationChart', 'No valid data for selected variables');
        return;
    }
    
    // Calculate correlation
    const xValues = validData.map(d => parseFloat(d[xVar]));
    const yValues = validData.map(d => parseFloat(d[yVar]));
    const correlation = calculateCorrelation(xValues, yValues);
    const rSquared = correlation * correlation;
    
    // Update stats
    $('#correlationValue').text(correlation.toFixed(3));
    $('#rSquaredValue').text(rSquared.toFixed(3));
    
    // Create scatter plot
    const trace = {
        x: xValues,
        y: yValues,
        mode: 'markers',
        type: 'scatter',
        name: 'Data Points',
        marker: {
            size: 8,
            opacity: 0.6,
            color: validData.map(d => getColorForCategory(d[colorVar]))
        },
        text: validData.map(d => 
            `${colorVar}: ${d[colorVar]}<br>${xVar}: ${d[xVar]}<br>${yVar}: ${d[yVar]}`
        ),
        hoverinfo: 'text'
    };
    
    // Add trend line
    const trendLine = calculateTrendLine(xValues, yValues);
    const traceLine = {
        x: [Math.min(...xValues), Math.max(...xValues)],
        y: [trendLine.slope * Math.min(...xValues) + trendLine.intercept, 
            trendLine.slope * Math.max(...xValues) + trendLine.intercept],
        mode: 'lines',
        type: 'scatter',
        name: `Trend Line (R²=${rSquared.toFixed(3)})`,
        line: {
            color: '#e74c3c',
            width: 2
        }
    };
    
    const layout = {
        title: `${yVar} vs ${xVar} by ${colorVar}`,
        xaxis: { title: xVar.replace('_', ' ') },
        yaxis: { title: yVar.replace('_', ' ') },
        height: 400,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        showlegend: true
    };
    
    Plotly.newPlot('correlationChart', [trace, traceLine], layout);
}

function loadTimeSeriesAnalysis() {
    if (!analysisData || analysisData.length === 0) {
        console.warn("No data available for time series analysis");
        return;
    }
    
    const timeVar = $('#timeVariable').val();
    const metric = $('#timeMetric').val();
    const aggregation = $('#timeAggregation').val();
    const groupBy = $('#timeGroup').val();
    
    console.log(`Loading time series: ${metric} by ${timeVar} grouped by ${groupBy}`);
    
    // Check if time variable exists in data
    if (!analysisData[0].hasOwnProperty(timeVar)) {
        showChartError('timeSeriesChart', `Time variable "${timeVar}" not found in data`);
        return;
    }
    
    // Group data
    const groupedData = {};
    analysisData.forEach(d => {
        const timeValue = d[timeVar];
        const metricValue = parseFloat(d[metric]);
        const groupValue = groupBy === 'none' ? 'All' : d[groupBy];
        
        if (timeValue !== null && !isNaN(metricValue)) {
            if (!groupedData[groupValue]) {
                groupedData[groupValue] = {};
            }
            if (!groupedData[groupValue][timeValue]) {
                groupedData[groupValue][timeValue] = [];
            }
            groupedData[groupValue][timeValue].push(metricValue);
        }
    });
    
    // Create traces for each group
    const traces = [];
    Object.keys(groupedData).forEach(group => {
        const timeValues = Object.keys(groupedData[group]).sort();
        const aggregatedValues = timeValues.map(time => {
            const values = groupedData[group][time];
            switch(aggregation) {
                case 'mean':
                    return values.reduce((a, b) => a + b, 0) / values.length;
                case 'sum':
                    return values.reduce((a, b) => a + b, 0);
                case 'median':
                    return values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
                case 'std':
                    const mean = values.reduce((a, b) => a + b, 0) / values.length;
                    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
                    return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / values.length);
                default:
                    return values.reduce((a, b) => a + b, 0) / values.length;
            }
        });
        
        traces.push({
            x: timeValues,
            y: aggregatedValues,
            mode: 'lines+markers',
            type: 'scatter',
            name: group,
            line: { width: 2 }
        });
    });
    
    const layout = {
        title: `${aggregation.charAt(0).toUpperCase() + aggregation.slice(1)} ${metric} by ${timeVar}`,
        xaxis: { title: timeVar.replace('_', ' ') },
        yaxis: { title: metric.replace('_', ' ') },
        height: 400,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        showlegend: true
    };
    
    Plotly.newPlot('timeSeriesChart', traces, layout);
}

function loadDistributionAnalysis() {
    if (!analysisData || analysisData.length === 0) {
        console.warn("No data available for distribution analysis");
        return;
    }
    
    const variable = $('#distributionVar').val();
    const chartType = $('#distributionType').val();
    
    console.log(`Loading ${chartType} distribution for ${variable}`);
    
    // Filter out invalid data
    const values = analysisData
        .map(d => parseFloat(d[variable]))
        .filter(v => !isNaN(v));
    
    if (values.length === 0) {
        showChartError('distributionChart', `No valid data for ${variable}`);
        return;
    }
    
    // Calculate statistics
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sorted = [...values].sort((a, b) => a - b);
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    // Calculate skewness
    const skewness = calculateSkewness(values, mean, stdDev);
    const kurtosis = calculateKurtosis(values, mean, stdDev);
    
    // Update stats
    $('#skewnessValue').text(skewness.toFixed(3));
    $('#kurtosisValue').text(kurtosis.toFixed(3));
    
    // Create chart based on type
    let trace;
    switch(chartType) {
        case 'histogram':
            trace = {
                x: values,
                type: 'histogram',
                name: 'Distribution',
                nbinsx: 30,
                marker: {
                    color: '#3498db',
                    opacity: 0.7
                }
            };
            break;
            
        case 'box':
            trace = {
                y: values,
                type: 'box',
                name: variable,
                boxpoints: 'outliers',
                marker: {
                    color: '#2ecc71'
                }
            };
            break;
            
        case 'violin':
            trace = {
                y: values,
                type: 'violin',
                name: variable,
                box: {
                    visible: true
                },
                meanline: {
                    visible: true
                },
                marker: {
                    color: '#e74c3c'
                }
            };
            break;
            
        case 'density':
            trace = {
                x: values,
                type: 'histogram',
                histnorm: 'probability density',
                name: 'Density',
                marker: {
                    color: '#9b59b6',
                    opacity: 0.6
                }
            };
            break;
    }
    
    const layout = {
        title: `${chartType.charAt(0).toUpperCase() + chartType.slice(1)} of ${variable}`,
        xaxis: { title: variable.replace('_', ' ') },
        yaxis: { title: chartType === 'box' || chartType === 'violin' ? variable.replace('_', ' ') : 'Frequency' },
        height: 300,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot('distributionChart', [trace], layout);
}

function loadComparisonAnalysis() {
    if (!analysisData || analysisData.length === 0) {
        console.warn("No data available for comparison analysis");
        return;
    }
    
    const compareBy = $('#compareBy').val();
    const metric = $('#compareMetric').val();
    
    console.log(`Loading comparison: ${metric} by ${compareBy}`);
    
    // Group data
    const groupedData = {};
    analysisData.forEach(d => {
        const group = d[compareBy] || 'Unknown';
        const value = parseFloat(d[metric]);
        
        if (!isNaN(value)) {
            if (!groupedData[group]) {
                groupedData[group] = [];
            }
            groupedData[group].push(value);
        }
    });
    
    // Prepare data for box plot
    const traces = Object.keys(groupedData).map(group => {
        return {
            y: groupedData[group],
            type: 'box',
            name: group,
            boxpoints: 'all',
            jitter: 0.3,
            pointpos: -1.8,
            marker: {
                size: 4,
                opacity: 0.6
            }
        };
    });
    
    // Sort by median value
    traces.sort((a, b) => {
        const medianA = a.y.sort((x, y) => x - y)[Math.floor(a.y.length / 2)];
        const medianB = b.y.sort((x, y) => x - y)[Math.floor(b.y.length / 2)];
        return medianB - medianA; // Descending
    });
    
    const layout = {
        title: `${metric} Comparison by ${compareBy}`,
        yaxis: { title: metric.replace('_', ' ') },
        xaxis: { title: compareBy.replace('_', ' ') },
        height: 300,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        showlegend: true
    };
    
    Plotly.newPlot('comparisonChart', traces, layout);
}

function loadHeatmapAnalysis() {
    if (!analysisData || analysisData.length === 0) {
        console.warn("No data available for heatmap analysis");
        return;
    }
    
    const xVar = $('#heatmapX').val();
    const yVar = $('#heatmapY').val();
    const metric = $('#heatmapMetric').val();
    const aggregation = $('#heatmapAgg').val();
    
    console.log(`Loading heatmap: ${metric} by ${xVar} x ${yVar}`);
    
    // Get unique values
    const xValues = [...new Set(analysisData.map(d => d[xVar]))].sort();
    const yValues = [...new Set(analysisData.map(d => d[yVar]))].sort();
    
    // Initialize matrix
    const zMatrix = [];
    const textMatrix = [];
    
    for (let i = 0; i < yValues.length; i++) {
        zMatrix[i] = [];
        textMatrix[i] = [];
        
        for (let j = 0; j < xValues.length; j++) {
            const filtered = analysisData.filter(d => 
                d[xVar] === xValues[j] && d[yVar] === yValues[i]
            );
            
            let value = 0;
            let count = filtered.length;
            
            if (count > 0) {
                if (metric === 'count') {
                    value = count;
                } else {
                    const values = filtered.map(d => parseFloat(d[metric])).filter(v => !isNaN(v));
                    if (values.length > 0) {
                        switch(aggregation) {
                            case 'mean':
                                value = values.reduce((a, b) => a + b, 0) / values.length;
                                break;
                            case 'sum':
                                value = values.reduce((a, b) => a + b, 0);
                                break;
                            case 'median':
                                const sorted = values.sort((a, b) => a - b);
                                value = sorted[Math.floor(sorted.length / 2)];
                                break;
                        }
                    }
                }
            }
            
            zMatrix[i][j] = value;
            textMatrix[i][j] = `${xVar}: ${xValues[j]}<br>${yVar}: ${yValues[i]}<br>${metric}: ${value.toFixed(2)}<br>Count: ${count}`;
        }
    }
    
    const trace = {
        x: xValues,
        y: yValues,
        z: zMatrix,
        text: textMatrix,
        type: 'heatmap',
        colorscale: 'Viridis',
        hoverinfo: 'text'
    };
    
    const layout = {
        title: `${metric} by ${xVar} and ${yVar}`,
        xaxis: { title: xVar.replace('_', ' ') },
        yaxis: { title: yVar.replace('_', ' ') },
        height: 400,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot('heatmapChart', [trace], layout);
}

function loadStatisticalSummary() {
    if (!analysisData || analysisData.length === 0) {
        console.warn("No data available for statistical summary");
        return;
    }
    
    // Define numerical columns to analyze
    const numericalColumns = [
        'yield_kg_per_hectare',
        'soil_moisture_%',
        'temperature_C',
        'rainfall_mm',
        'NDVI_index',
        'soil_pH',
        'humidity_%',
        'sunlight_hours',
        'pesticide_usage_ml',
        'total_days'
    ];
    
    // Filter columns that exist in data
    const existingColumns = numericalColumns.filter(col => 
        analysisData[0].hasOwnProperty(col)
    );
    
    const tbody = $('#statisticalTable tbody');
    let html = '';
    
    existingColumns.forEach(column => {
        const values = analysisData
            .map(d => parseFloat(d[column]))
            .filter(v => !isNaN(v));
        
        if (values.length > 0) {
            const sorted = [...values].sort((a, b) => a - b);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
            const stdDev = Math.sqrt(variance);
            const min = sorted[0];
            const q1 = sorted[Math.floor(sorted.length * 0.25)];
            const median = sorted[Math.floor(sorted.length * 0.5)];
            const q3 = sorted[Math.floor(sorted.length * 0.75)];
            const max = sorted[sorted.length - 1];
            
            html += `
                <tr>
                    <td>${column.replace('_', ' ')}</td>
                    <td>${mean.toFixed(2)}</td>
                    <td>${stdDev.toFixed(2)}</td>
                    <td>${min.toFixed(2)}</td>
                    <td>${q1.toFixed(2)}</td>
                    <td>${median.toFixed(2)}</td>
                    <td>${q3.toFixed(2)}</td>
                    <td>${max.toFixed(2)}</td>
                </tr>
            `;
        }
    });
    
    tbody.html(html);
}

function runStatisticalTest() {
    if (!analysisData || analysisData.length === 0) {
        alert('No data available for statistical test');
        return;
    }
    
    const compareBy = $('#compareBy').val();
    const metric = $('#compareMetric').val();
    
    // Group data
    const groups = {};
    analysisData.forEach(d => {
        const group = d[compareBy] || 'Unknown';
        const value = parseFloat(d[metric]);
        
        if (!isNaN(value)) {
            if (!groups[group]) {
                groups[group] = [];
            }
            groups[group].push(value);
        }
    });
    
    // Perform simple ANOVA (simplified)
    const groupMeans = {};
    const overallMean = analysisData
        .map(d => parseFloat(d[metric]))
        .filter(v => !isNaN(v))
        .reduce((a, b) => a + b, 0) / analysisData.length;
    
    let ssBetween = 0;
    let ssWithin = 0;
    
    Object.keys(groups).forEach(group => {
        const values = groups[group];
        const groupMean = values.reduce((a, b) => a + b, 0) / values.length;
        groupMeans[group] = groupMean;
        
        // Sum of squares between groups
        ssBetween += values.length * Math.pow(groupMean - overallMean, 2);
        
        // Sum of squares within groups
        ssWithin += values.reduce((sum, value) => sum + Math.pow(value - groupMean, 2), 0);
    });
    
    const dfBetween = Object.keys(groups).length - 1;
    const dfWithin = analysisData.length - Object.keys(groups).length;
    const msBetween = ssBetween / dfBetween;
    const msWithin = ssWithin / dfWithin;
    const fValue = msBetween / msWithin;
    
    // Simple interpretation
    let interpretation = '';
    if (fValue > 3) { // Simplified threshold
        interpretation = 'Significant differences exist between groups (p < 0.05)';
    } else {
        interpretation = 'No significant differences between groups';
    }
    
    // Show results
    $('#testResults').html(`
        <div class="alert alert-info">
            <h6><i class="fas fa-chart-line me-2"></i> ANOVA Results</h6>
            <p class="mb-1">F-value: <strong>${fValue.toFixed(3)}</strong></p>
            <p class="mb-1">Between-group SS: ${ssBetween.toFixed(2)}</p>
            <p class="mb-1">Within-group SS: ${ssWithin.toFixed(2)}</p>
            <p class="mb-0"><strong>${interpretation}</strong></p>
        </div>
    `);
}

function downloadStatisticalReport() {
    // Create CSV content
    let csvContent = "Variable,Mean,Std Dev,Min,25%,Median,75%,Max\n";
    
    // Get table data
    $('#statisticalTable tbody tr').each(function() {
        const cells = $(this).find('td');
        const row = [];
        cells.each(function() {
            row.push($(this).text());
        });
        csvContent += row.join(',') + '\n';
    });
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `statistical_report_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Helper functions
function calculateCorrelation(x, y) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

function calculateTrendLine(x, y) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return { slope, intercept };
}

function calculateSkewness(values, mean, stdDev) {
    if (stdDev === 0) return 0;
    
    const n = values.length;
    const cubedDeviations = values.reduce((sum, val) => {
        const deviation = val - mean;
        return sum + Math.pow(deviation, 3);
    }, 0);
    
    return (cubedDeviations / n) / Math.pow(stdDev, 3);
}

function calculateKurtosis(values, mean, stdDev) {
    if (stdDev === 0) return 0;
    
    const n = values.length;
    const fourthDeviations = values.reduce((sum, val) => {
        const deviation = val - mean;
        return sum + Math.pow(deviation, 4);
    }, 0);
    
    return (fourthDeviations / n) / Math.pow(stdDev, 4) - 3;
}

function getColorForCategory(category) {
    // Simple hash function to generate consistent colors
    let hash = 0;
    const str = String(category);
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    const colors = [
        '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
        '#1abc9c', '#34495e', '#d35400', '#c0392b', '#16a085'
    ];
    
    return colors[Math.abs(hash) % colors.length];
}

function showAnalysisError(message) {
    const container = $('.container-fluid');
    container.prepend(`
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
}

function showChartError(chartId, message) {
    const container = document.getElementById(chartId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-warning m-3">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
}

// Initialize on page load
$(document).ready(function() {
    console.log("Analysis page loaded");
    initializeAnalysis();
});