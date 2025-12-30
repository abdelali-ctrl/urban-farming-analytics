// Predictions JavaScript - Complete fixed version

function initializePredictionForm() {
    console.log("Initializing prediction form...");
    
    // Setup form submission
    $('#predictionForm').on('submit', function(event) {
        event.preventDefault();
        console.log("Form submit triggered");
        predictYield();
    });
    
    // Setup quick fill buttons
    $('#fillOptimal').on('click', function() {
        fillOptimalValues();
    });
    
    $('#fillAverage').on('click', function() {
        fillAverageValues();
    });
    
    $('#fillPoor').on('click', function() {
        fillPoorValues();
    });
    
    console.log("Prediction form initialized");
}

// Quick fill functions
function fillOptimalValues() {
    $('#soil_moisture').val(55);
    $('#soil_pH').val(6.5);
    $('#temperature').val(25);
    $('#rainfall').val(600);
    $('#humidity').val(65);
    $('#sunlight_hours').val(8.5);
    $('#pesticide_usage').val(100);
    $('#NDVI_index').val(0.75);
    $('#total_days').val(120);
    $('#disease_status').val('No Disease');
    console.log("Filled optimal values");
}

function fillAverageValues() {
    $('#soil_moisture').val(50);
    $('#soil_pH').val(6.2);
    $('#temperature').val(22);
    $('#rainfall').val(500);
    $('#humidity').val(60);
    $('#sunlight_hours').val(7.5);
    $('#pesticide_usage').val(100);
    $('#NDVI_index').val(0.65);
    $('#total_days').val(120);
    $('#disease_status').val('No Disease');
    console.log("Filled average values");
}

function fillPoorValues() {
    $('#soil_moisture').val(35);
    $('#soil_pH').val(5.8);
    $('#temperature').val(18);
    $('#rainfall').val(300);
    $('#humidity').val(50);
    $('#sunlight_hours').val(6);
    $('#pesticide_usage').val(100);
    $('#NDVI_index').val(0.45);
    $('#total_days').val(120);
    $('#disease_status').val('Moderate');
    console.log("Filled poor values");
}

function predictYield() {
    console.log("Starting prediction...");
    
    // Collect form data
    const formData = {
        soil_moisture: parseFloat($('#soil_moisture').val()) || 50,
        soil_pH: parseFloat($('#soil_pH').val()) || 6.5,
        temperature: parseFloat($('#temperature').val()) || 25,
        rainfall: parseFloat($('#rainfall').val()) || 500,
        humidity: parseFloat($('#humidity').val()) || 65,
        sunlight_hours: parseFloat($('#sunlight_hours').val()) || 8,
        pesticide_usage: parseFloat($('#pesticide_usage').val()) || 100,
        NDVI_index: parseFloat($('#NDVI_index').val()) || 0.7,
        total_days: parseInt($('#total_days').val()) || 120,
        region: $('#region').val() || 'North',
        crop_type: $('#crop_type').val() || 'Wheat',
        irrigation_type: $('#irrigation_type').val() || 'Drip',
        fertilizer_type: $('#fertilizer_type').val() || 'Organic',
        disease_status: $('#disease_status').val() || 'No Disease'
    };
    
    console.log("Form data to send:", formData);
    
    // Validate
    if (!validateForm(formData)) {
        alert('Please fill in all required fields correctly');
        return;
    }
    
    // Show loading
    const submitBtn = $('#predictBtn');
    const originalText = submitBtn.html();
    submitBtn.prop('disabled', true);
    submitBtn.html('<i class="fas fa-spinner fa-spin me-2"></i> Predicting...');
    
    // Send request
    $.ajax({
        url: '/api/predict',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(formData),
        success: function(response) {
            console.log("Prediction response received:", response);
            
            if (response.success) {
                displayPredictionResults(response);
            } else {
                showError('Prediction failed: ' + (response.error || 'Unknown error'));
            }
        },
        error: function(xhr, status, error) {
            console.error("AJAX error:", xhr.responseText);
            showError('Server error: ' + error + '. Check console for details.');
        },
        complete: function() {
            // Restore button
            submitBtn.prop('disabled', false);
            submitBtn.html(originalText);
        }
    });
}

function validateForm(data) {
    let isValid = true;
    
    // Check numeric ranges
    if (data.soil_moisture < 0 || data.soil_moisture > 100) {
        $('#soil_moisture').addClass('is-invalid');
        isValid = false;
    } else {
        $('#soil_moisture').removeClass('is-invalid');
    }
    
    if (data.soil_pH < 4 || data.soil_pH > 9) {
        $('#soil_pH').addClass('is-invalid');
        isValid = false;
    } else {
        $('#soil_pH').removeClass('is-invalid');
    }
    
    if (data.NDVI_index < 0 || data.NDVI_index > 1) {
        $('#NDVI_index').addClass('is-invalid');
        isValid = false;
    } else {
        $('#NDVI_index').removeClass('is-invalid');
    }
    
    return isValid;
}

function displayPredictionResults(response) {
    console.log("Displaying prediction results:", response);
    
    const prediction = response.prediction || 0;
    const confidence = response.confidence || 0.5;
    
    // Update prediction value
    $('#predictionValue').text(prediction.toFixed(0) + ' kg/ha');
    
    // Update confidence
    const confidencePercent = Math.round(confidence * 100);
    $('#confidenceBadge').text('Confidence: ' + confidencePercent + '%');
    
    // Color code confidence
    updateConfidenceBadge(confidence);
    
    // Display recommendations
    displayRecommendations(response.recommendations || []);
    
    // Display similar farms
    displaySimilarFarms(response.similar_farms || []);
    
    // Update performance comparison and target achievement
    updatePerformanceComparison(prediction, response.crop_type || $('#crop_type').val());
    updateTargetAchievement(prediction, response.region || $('#region').val());
    
    // Show results
    $('#resultCard').fadeIn();
    $('#recommendationsCard').fadeIn();
    
    // Add to history
    addToHistory(response);
}

// Helper functions that are missing
function formatNumber(num) {
    if (num === undefined || num === null || isNaN(num)) return 'N/A';
    
    const number = parseFloat(num);
    if (Number.isInteger(number)) {
        return number.toLocaleString('en-US');
    } else {
        return number.toLocaleString('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 2
        });
    }
}

function getYieldClass(yieldValue) {
    if (!yieldValue || isNaN(yieldValue)) return 'text-muted';
    
    const yieldNum = parseFloat(yieldValue);
    if (yieldNum >= 5000) return 'text-success';
    if (yieldNum >= 3000) return 'text-warning';
    return 'text-danger';
}

function getDiseaseClass(diseaseStatus) {
    if (!diseaseStatus) return 'bg-secondary';
    
    const status = diseaseStatus.toString().toLowerCase();
    if (status.includes('no disease') || status.includes('healthy')) return 'bg-success';
    if (status.includes('mild')) return 'bg-info';
    if (status.includes('moderate')) return 'bg-warning';
    if (status.includes('severe')) return 'bg-danger';
    return 'bg-secondary';
}

function initializeSensitivityAnalysis() {
    console.log("Initializing sensitivity analysis...");
    
    // Setup sliders
    const sliders = [
        { id: 'moistureSensitivity', param: 'soil_moisture', min: 20, max: 80, defaultValue: 50 },
        { id: 'tempSensitivity', param: 'temperature', min: 10, max: 40, defaultValue: 25 },
        { id: 'rainSensitivity', param: 'rainfall', min: 200, max: 1200, defaultValue: 500 },
        { id: 'ndviSensitivity', param: 'NDVI_index', min: 0.3, max: 0.9, defaultValue: 0.7, step: 0.01 }
    ];
    
    sliders.forEach(slider => {
        const element = $(`#${slider.id}`);
        const valueElement = $(`#${slider.id.replace('Sensitivity', 'Value')}`);
        const yieldElement = $(`#${slider.id.replace('Sensitivity', 'Yield')}`);
        
        // Set initial value
        element.val(slider.defaultValue);
        valueElement.text(slider.defaultValue + (slider.id === 'tempSensitivity' ? '°C' : 
                               slider.id === 'rainSensitivity' ? 'mm' : 
                               slider.id === 'moistureSensitivity' ? '%' : ''));
        
        // On slider change
        element.on('input', function() {
            const value = parseFloat($(this).val());
            valueElement.text(value + (slider.id === 'tempSensitivity' ? '°C' : 
                                     slider.id === 'rainSensitivity' ? 'mm' : 
                                     slider.id === 'moistureSensitivity' ? '%' : ''));
            
            // Run sensitivity analysis
            runSensitivityAnalysis(slider.param, value);
        });
    });
    
    // Initialize chart
    initializeSensitivityChart();
}

function initializeSensitivityChart() {
    const ctx = document.getElementById('sensitivityChart').getContext('2d');
    window.sensitivityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['-30%', '-20%', '-10%', 'Current', '+10%', '+20%', '+30%'],
            datasets: [{
                label: 'Predicted Yield',
                data: [],
                borderColor: '#2ecc71',
                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Yield: ${context.parsed.y.toFixed(0)} kg/ha`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Yield (kg/ha)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Parameter Variation'
                    }
                }
            }
        }
    });
}

function runSensitivityAnalysis(param, baseValue) {
    console.log(`Running sensitivity analysis for ${param} with base value ${baseValue}`);
    
    // Get current form values
    const formData = getCurrentFormData();
    
    // Create variations (±30%)
    const variations = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3];
    const predictions = [];
    
    // Show loading
    $(`#${param}Yield`).html('<i class="fas fa-spinner fa-spin"></i>');
    
    // Make prediction for each variation
    let completed = 0;
    variations.forEach((variation, index) => {
        const testData = { ...formData };
        const variationValue = baseValue * (1 + variation);
        testData[param] = variationValue;
        
        // For percentage values, ensure they stay in range
        if (param === 'soil_moisture') {
            testData[param] = Math.max(0, Math.min(100, variationValue));
        } else if (param === 'NDVI_index') {
            testData[param] = Math.max(0, Math.min(1, variationValue));
        }
        
        // Send prediction request
        setTimeout(() => {
            $.ajax({
                url: '/api/predict',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(testData),
                success: function(response) {
                    if (response.success) {
                        predictions[index] = response.prediction;
                        completed++;
                        
                        // Update yield for current value
                        if (variation === 0) {
                            $(`#${param}Yield`).text(response.prediction.toFixed(0) + ' kg/ha');
                        }
                        
                        // Update chart when all predictions are done
                        if (completed === variations.length) {
                            updateSensitivityChart(predictions);
                        }
                    }
                },
                error: function() {
                    predictions[index] = 0;
                    completed++;
                }
            });
        }, index * 100); // Stagger requests
    });
}

function getCurrentFormData() {
    return {
        soil_moisture: parseFloat($('#soil_moisture').val()) || 50,
        soil_pH: parseFloat($('#soil_pH').val()) || 6.5,
        temperature: parseFloat($('#temperature').val()) || 25,
        rainfall: parseFloat($('#rainfall').val()) || 500,
        humidity: parseFloat($('#humidity').val()) || 65,
        sunlight_hours: parseFloat($('#sunlight_hours').val()) || 8,
        pesticide_usage: parseFloat($('#pesticide_usage').val()) || 100,
        NDVI_index: parseFloat($('#NDVI_index').val()) || 0.7,
        total_days: parseInt($('#total_days').val()) || 120,
        region: $('#region').val() || 'North',
        crop_type: $('#crop_type').val() || 'Wheat',
        irrigation_type: $('#irrigation_type').val() || 'Drip',
        fertilizer_type: $('#fertilizer_type').val() || 'Organic',
        disease_status: $('#disease_status').val() || 'No Disease'
    };
}

function updateSensitivityChart(predictions) {
    if (window.sensitivityChart) {
        window.sensitivityChart.data.datasets[0].data = predictions;
        window.sensitivityChart.update();
    }
}

function updateConfidenceBadge(confidence) {
    const badge = $('#confidenceBadge');
    if (confidence > 0.8) {
        badge.removeClass('bg-warning bg-danger').addClass('bg-success');
    } else if (confidence > 0.6) {
        badge.removeClass('bg-success bg-danger').addClass('bg-warning');
    } else {
        badge.removeClass('bg-success bg-warning').addClass('bg-danger');
    }
}

function updatePerformanceComparison(prediction, cropType) {
    // Get average yield for this crop from the server
    $.ajax({
        url: '/api/data/stats',
        type: 'GET',
        success: function(response) {
            if (response.success && response.yield_statistics) {
                const cropAvg = response.yield_statistics.mean || 0;
                const diff = prediction - cropAvg;
                const diffPercent = cropAvg > 0 ? (diff / cropAvg * 100) : 0;
                
                let performanceText = '';
                let performanceClass = '';
                
                if (diffPercent > 20) {
                    performanceText = `Excellent (${diffPercent.toFixed(1)}% above average)`;
                    performanceClass = 'text-success';
                } else if (diffPercent > 0) {
                    performanceText = `Good (${diffPercent.toFixed(1)}% above average)`;
                    performanceClass = 'text-success';
                } else if (diffPercent > -10) {
                    performanceText = `Average (${Math.abs(diffPercent).toFixed(1)}% below average)`;
                    performanceClass = 'text-warning';
                } else {
                    performanceText = `Below Average (${Math.abs(diffPercent).toFixed(1)}% below average)`;
                    performanceClass = 'text-danger';
                }
                
                // Create simple chart
                const chartData = [prediction, cropAvg];
                const maxValue = Math.max(prediction, cropAvg) * 1.1;
                
                const trace = {
                    x: ['Your Prediction', 'Crop Average'],
                    y: [prediction, cropAvg],
                    type: 'bar',
                    marker: {
                        color: ['#2ecc71', '#3498db']
                    }
                };
                
                const layout = {
                    height: 150,
                    margin: { t: 10, b: 30, l: 40, r: 10 },
                    showlegend: false,
                    yaxis: { 
                        title: 'Yield (kg/ha)',
                        range: [0, maxValue]
                    },
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)'
                };
                
                Plotly.newPlot('performanceChart', [trace], layout);
                
                // Add performance text
                $('#performanceChart').after(`
                    <div class="text-center mt-2 ${performanceClass}">
                        <small><i class="fas fa-chart-line me-1"></i> ${performanceText}</small>
                    </div>
                `);
            }
        }
    });
}

function updateTargetAchievement(prediction, region) {
    // Get regional average yield
    $.ajax({
        url: '/api/data/filter',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ region: region }),
        success: function(response) {
            if (response.success && response.average_yield > 0) {
                const regionalAvg = response.average_yield;
                const achievementPercent = Math.min((prediction / regionalAvg) * 100, 150);
                
                // Update percentage
                $('#targetPercentage').text(Math.round(achievementPercent) + '%');
                
                // Update circular progress
                const circle = $('#progressCircle')[0];
                const radius = 45;
                const circumference = 2 * Math.PI * radius;
                const offset = circumference - (achievementPercent / 100) * circumference;
                circle.style.strokeDashoffset = offset;
                
                // Update text
                let achievementText = '';
                if (achievementPercent >= 120) {
                    achievementText = 'Excellent - Well above regional average';
                    $('#progressCircle').css('stroke', '#2ecc71');
                } else if (achievementPercent >= 100) {
                    achievementText = 'Good - Above regional average';
                    $('#progressCircle').css('stroke', '#2ecc71');
                } else if (achievementPercent >= 80) {
                    achievementText = 'Average - Near regional average';
                    $('#progressCircle').css('stroke', '#f39c12');
                } else {
                    achievementText = 'Below regional average';
                    $('#progressCircle').css('stroke', '#e74c3c');
                }
                
                $('#targetText').text(achievementText);
            } else {
                // Fallback to national average
                $.ajax({
                    url: '/api/data/stats',
                    type: 'GET',
                    success: function(statsResponse) {
                        if (statsResponse.success && statsResponse.yield_statistics) {
                            const nationalAvg = statsResponse.yield_statistics.mean || 0;
                            const achievementPercent = Math.min((prediction / nationalAvg) * 100, 150);
                            
                            $('#targetPercentage').text(Math.round(achievementPercent) + '%');
                            $('#targetText').text('of national average');
                        }
                    }
                });
            }
        }
    });
}

function displayRecommendations(recommendations) {
    const container = $('#recommendationsList');
    
    if (!recommendations || recommendations.length === 0) {
        container.html(`
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                No specific recommendations needed. Your farm parameters look good!
            </div>
        `);
        return;
    }
    
    let html = '';
    recommendations.forEach((rec, index) => {
        const priority = rec.priority || 'Medium';
        const priorityClass = priority.toLowerCase();
        
        html += `
            <div class="recommendation-item ${priorityClass} mb-3">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="me-3">
                        <h6 class="mb-1">
                            <i class="fas fa-lightbulb me-2"></i>
                            ${rec.category || 'Recommendation'}
                        </h6>
                        <p class="mb-1 small">${rec.message || 'No message provided'}</p>
                    </div>
                    <span class="badge ${getPriorityBadgeClass(priority)}">
                        ${priority} Priority
                    </span>
                </div>
                ${rec.expected_impact ? `
                    <div class="mt-1">
                        <small class="text-muted">
                            <i class="fas fa-chart-line me-1"></i>
                            Expected impact: ${rec.expected_impact}
                        </small>
                    </div>
                ` : ''}
            </div>
        `;
    });
    
    container.html(html);
}

function getPriorityBadgeClass(priority) {
    if (!priority) return 'bg-secondary';
    
    const priorityLower = priority.toLowerCase();
    switch(priorityLower) {
        case 'high': return 'bg-danger';
        case 'medium': return 'bg-warning';
        case 'low': return 'bg-info';
        default: return 'bg-secondary';
    }
}

function displaySimilarFarms(similarFarms) {
    const tbody = $('#similarFarmsTable tbody');
    
    if (!similarFarms || similarFarms.length === 0) {
        tbody.html(`
            <tr>
                <td colspan="4" class="text-center py-4">
                    <div class="text-muted">
                        <i class="fas fa-search me-2"></i>
                        No similar farms found with exact matching criteria
                    </div>
                    <small class="text-muted">
                        Try broadening your search criteria or check if similar farms exist in the database
                    </small>
                </td>
            </tr>
        `);
        return;
    }
    
    let html = '';
    similarFarms.forEach((farm, index) => {
        const similarity = farm.similarity_score || 0.5;
        const similarityPercent = Math.round(similarity * 100);
        
        html += `
            <tr>
                <td class="fw-bold">${farm.farm_id || farm.id || `Farm ${index + 1}`}</td>
                <td>
                    <div class="fw-bold ${getYieldClass(farm.yield_kg_per_hectare)}">
                        ${formatNumber(farm.yield_kg_per_hectare)} kg/ha
                    </div>
                    <small class="text-muted">${farm.crop_type || 'Crop'}</small>
                </td>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="progress flex-grow-1 me-2" style="height: 8px;">
                            <div class="progress-bar ${similarity > 0.7 ? 'bg-success' : similarity > 0.5 ? 'bg-warning' : 'bg-info'}" 
                                 style="width: ${similarityPercent}%"></div>
                        </div>
                        <small class="fw-bold">${similarityPercent}%</small>
                    </div>
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-info" onclick="viewFarmDetails(${farm.farm_id || farm.id || index})">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    });
    
    tbody.html(html);
}

function addToHistory(response) {
    // Get existing history or create new array
    const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
    
    // Add new prediction
    const historyItem = {
        timestamp: new Date().toLocaleString(),
        crop: response.input_data?.crop_type || $('#crop_type').val(),
        predicted: response.prediction.toFixed(0),
        confidence: Math.round((response.confidence || 0.5) * 100),
        model: response.model_used || 'Unknown'
    };
    
    history.unshift(historyItem);
    
    // Keep only last 10 items
    if (history.length > 10) {
        history.pop();
    }
    
    // Save back to localStorage
    localStorage.setItem('predictionHistory', JSON.stringify(history));
    
    // Update display
    updateHistoryDisplay(history);
}

function updateHistoryDisplay(history) {
    const tbody = $('#predictionHistory tbody');
    
    if (!history || history.length === 0) {
        tbody.html(`
            <tr>
                <td colspan="4" class="text-center text-muted">
                    No prediction history yet
                </td>
            </tr>
        `);
        return;
    }
    
    let html = '';
    history.forEach((item, index) => {
        html += `
            <tr>
                <td><small>${item.timestamp}</small></td>
                <td>${item.crop}</td>
                <td>${item.predicted} kg/ha</td>
                <td>
                    <span class="badge ${item.confidence > 70 ? 'bg-success' : 
                                        item.confidence > 50 ? 'bg-warning' : 'bg-danger'}">
                        ${item.confidence}%
                    </span>
                </td>
            </tr>
        `;
    });
    
    tbody.html(html);
}

function showError(message) {
    // Remove existing alerts
    $('.alert-danger').remove();
    
    // Show error alert
    $('.container-fluid').prepend(`
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <i class="fas fa-exclamation-circle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        $('.alert-danger').alert('close');
    }, 5000);
}


// Load history on page load
function loadHistory() {
    const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
    updateHistoryDisplay(history);
}

function viewFarmDetails(farmId) {
    // This could open a modal or redirect to a farm details page
    alert(`Viewing details for farm ${farmId}. This feature can be extended to show more details.`);
    
    // Example: You could fetch farm details from the server
    // $.ajax({
    //     url: `/api/data/farm/${farmId}`,
    //     type: 'GET',
    //     success: function(farmData) {
    //         // Display farm details in a modal
    //         showFarmModal(farmData);
    //     }
    // });
}

// Update the document ready function:
$(document).ready(function() {
    console.log("Document ready, initializing predictions...");
    initializePredictionForm();
    initializeSensitivityAnalysis(); // Add this line
    loadHistory();
    
    // Test if jQuery is working
    console.log("jQuery version:", $.fn.jquery);
    
    // Test if form exists
    console.log("Form exists:", $('#predictionForm').length > 0);
});