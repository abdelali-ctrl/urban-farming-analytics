// dashboard.js - Complete working version with improved validation

let currentFilters = {};
let dataStats = {}; // To store data statistics for validation

function initializeDashboard() {
    console.log("Dashboard initialized successfully");
    
    // Setup event listeners first
    setupEventListeners();
    
    // Load filter options
    loadFilterOptions();
    
    // Load data statistics for validation
    loadDataStats();
    
    // Load initial data with no filters
    loadFilteredData({});
    
    // Load initial charts after a short delay
    setTimeout(loadInitialCharts, 500);
}

function setupEventListeners() {
    console.log("Setting up event listeners");
    
    // Apply filters button
    $('#applyFilters').off('click').on('click', function() {
        console.log("Apply filters clicked");
        applyFilters();
    });
    
    // Reset filters button
    $('#resetFilters').off('click').on('click', function() {
        console.log("Reset filters clicked");
        resetFilters();
    });
    
    // Export data button
    $('#exportData').off('click').on('click', function() {
        console.log("Export data clicked");
        exportData();
    });
    
    // Quick filter buttons (if they exist)
    $('[id^="quickFilter"]').off('click').on('click', function() {
        const filterType = $(this).data('filter');
        applyQuickFilter(filterType);
    });
    
    // Add tooltips to yield inputs
    $('#minYield, #maxYield').tooltip({
        title: 'Typical range: 1000-8000 kg/ha',
        placement: 'top'
    });
}

function loadFilterOptions() {
    console.log("Loading filter options...");
    
    $.ajax({
        url: '/api/data/filters',
        type: 'GET',
        success: function(response) {
            console.log("Filter options API response:", response);
            
            if (response && response.success && response.filters) {
                // Populate dropdowns
                populateDropdown('#regionFilter', response.filters.regions || []);
                populateDropdown('#cropFilter', response.filters.crop_types || []);
                populateDropdown('#irrigationFilter', response.filters.irrigation_types || []);
                
                console.log("✓ Filter options loaded successfully");
            } else {
                console.error("Failed to load filter options:", response);
                showError('Failed to load filter options. Please refresh the page.');
            }
        },
        error: function(xhr, status, error) {
            console.error("Error loading filter options:", error);
            showError('Error loading filter options: ' + error);
        }
    });
}

function loadDataStats() {
    console.log("Loading data statistics for validation...");
    
    $.ajax({
        url: '/api/data/stats',
        type: 'GET',
        success: function(response) {
            console.log("Data stats response:", response);
            
            if (response && response.success && response.yield_statistics) {
                dataStats = response.yield_statistics;
                console.log("Data statistics loaded:", dataStats);
                
                // Update yield input placeholders based on actual data
                const minYield = Math.round(dataStats.min);
                const maxYield = Math.round(dataStats.max);
                
                $('#minYield').attr('placeholder', `Min (${minYield})`);
                $('#maxYield').attr('placeholder', `Max (${maxYield})`);
                
                // Update tooltips with actual data range
                $('#minYield, #maxYield').tooltip('dispose').tooltip({
                    title: `Actual range: ${minYield.toLocaleString()} - ${maxYield.toLocaleString()} kg/ha`,
                    placement: 'top'
                });
                
            } else {
                console.warn("Failed to load data statistics, using defaults");
                dataStats = {
                    min: 1000,
                    max: 8000,
                    mean: 4000
                };
            }
        },
        error: function(error) {
            console.warn("Error loading data stats:", error);
            dataStats = {
                min: 1000,
                max: 8000,
                mean: 4000
            };
        }
    });
}

function populateDropdown(selector, options) {
    const dropdown = $(selector);
    if (dropdown.length === 0) {
        console.error("Dropdown not found:", selector);
        return;
    }
    
    dropdown.empty();
    dropdown.append('<option value="">All</option>');
    
    if (options && Array.isArray(options)) {
        // Sort and filter options
        const sortedOptions = options
            .filter(opt => opt && opt.toString().trim() !== '')
            .sort();
        
        sortedOptions.forEach(function(option) {
            dropdown.append(`<option value="${option}">${option}</option>`);
        });
        
        console.log(`✓ Populated ${selector} with ${sortedOptions.length} options`);
    } else {
        console.warn(`No valid options for ${selector}`);
    }
}

function applyFilters() {
    console.log("Applying filters...");
    
    // Get filter values
    const filters = {
        region: $('#regionFilter').val() || '',
        crop_type: $('#cropFilter').val() || '',
        irrigation_type: $('#irrigationFilter').val() || '',
        min_yield: $('#minYield').val() || '',
        max_yield: $('#maxYield').val() || ''
    };
    
    console.log("Current filters:", filters);
    
    // Validate filters before applying
    if (!validateFilters(filters)) {
        return;
    }
    
    // Store filters globally
    currentFilters = filters;
    
    // Update filter summary
    updateFilterSummary(filters);
    
    // Load data with filters
    loadFilteredData(filters);
    
    // Update charts that use filters
    updateFilteredCharts();
}

function validateFilters(filters) {
    // Validate yield ranges
    if (filters.min_yield || filters.max_yield) {
        const min = parseFloat(filters.min_yield) || 0;
        const max = parseFloat(filters.max_yield) || Infinity;
        
        // Check if numbers are valid
        if ((filters.min_yield && isNaN(min)) || (filters.max_yield && isNaN(max))) {
            showError('Please enter valid numbers for yield range');
            return false;
        }
        
        // Check if min is greater than max
        if (min > max) {
            showError('Minimum yield cannot be greater than maximum yield');
            return false;
        }
        
        // Check for unrealistic low yields
        const dataMin = dataStats.min || 1000;
        if (min > 0 && min < dataMin * 0.5) { // Less than 50% of actual minimum
            if (!confirm(`Warning: Minimum yield (${min} kg/ha) is very low compared to actual data (min: ${dataMin} kg/ha). Continue anyway?`)) {
                return false;
            }
        }
        
        // Check for unrealistic high maximum
        const dataMax = dataStats.max || 8000;
        if (max < Infinity && max > dataMax * 1.5) { // More than 150% of actual maximum
            if (!confirm(`Warning: Maximum yield (${max} kg/ha) is very high compared to actual data (max: ${dataMax} kg/ha). Continue anyway?`)) {
                return false;
            }
        }
        
        // Check if range is too narrow
        if (min > 0 && max < Infinity && (max - min) < 100) {
            if (!confirm(`Warning: Yield range is very narrow (${max - min} kg/ha). This might return no results. Continue anyway?`)) {
                return false;
            }
        }
    }
    
    // Check for very specific filter combinations
    const filterCount = Object.values(filters).filter(v => v && v !== '').length;
    if (filterCount >= 3) {
        if (!confirm("You're applying multiple filters. This might return no results. Continue anyway?")) {
            return false;
        }
    }
    
    return true;
}

function updateFilterSummary(filters) {
    const summaryElements = [];
    
    if (filters.region) summaryElements.push(`Region: ${filters.region}`);
    if (filters.crop_type) summaryElements.push(`Crop: ${filters.crop_type}`);
    if (filters.irrigation_type) summaryElements.push(`Irrigation: ${filters.irrigation_type}`);
    if (filters.min_yield) summaryElements.push(`Min Yield: ${formatNumber(filters.min_yield)} kg/ha`);
    if (filters.max_yield) summaryElements.push(`Max Yield: ${formatNumber(filters.max_yield)} kg/ha`);
    
    const summaryText = summaryElements.length > 0 
        ? `Active filters: ${summaryElements.join(', ')}` 
        : 'No filters applied';
    
    const summaryElement = $('#filterSummary');
    summaryElement.text(summaryText);
    
    // Add warning if filters seem too restrictive
    if (summaryElements.length >= 3) {
        summaryElement.removeClass('text-muted').addClass('text-warning');
    } else if (summaryElements.length > 0) {
        summaryElement.removeClass('text-warning').addClass('text-info');
    } else {
        summaryElement.removeClass('text-warning text-info').addClass('text-muted');
    }
}

function resetFilters() {
    console.log("Resetting filters...");
    
    // Clear all filter inputs
    $('#regionFilter').val('');
    $('#cropFilter').val('');
    $('#irrigationFilter').val('');
    $('#minYield').val('');
    $('#maxYield').val('');
    
    // Reset current filters
    currentFilters = {};
    
    // Update filter summary
    updateFilterSummary({});
    
    // Load data without filters
    loadFilteredData({});
    
    // Reset charts
    updateFilteredCharts();
    
    showInfo('Filters reset successfully');
}

function loadFilteredData(filters) {
    console.log("Loading filtered data with filters:", filters);
    
    // Debug: Show what we're sending
    const jsonData = JSON.stringify(filters);
    console.log("Sending JSON data:", jsonData);
    
    // Show loading state in table
    $('#dataTable tbody').html(`
        <tr>
            <td colspan="8" class="text-center py-4">
                <div class="spinner-border spinner-border-sm text-success" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span class="ms-2">Loading farm data...</span>
            </td>
        </tr>
    `);
    
    $.ajax({
        url: '/api/data/filter',
        type: 'POST',
        contentType: 'application/json',
        data: jsonData,
        success: function(response) {
            console.log("Filter API response:", response);
            
            if (response && response.success) {
                // Debug: Check first record structure if available
                if (response.data && response.data.length > 0) {
                    console.log("First record sample:", response.data[0]);
                }
                
                // Update data table
                updateDataTable(response.data || []);
                
                // Update statistics
                updateSummaryStats(response);
                
                // Show message if no data
                if (!response.data || response.data.length === 0) {
                    handleNoData(filters);
                } else {
                    // Clear any previous error messages
                    $('.alert-info, .alert-warning').alert('close');
                }
                
                console.log(`✓ Loaded ${response.data ? response.data.length : 0} records`);
            } else {
                console.error("API returned error:", response);
                showError('Server error: ' + (response.error || 'Unknown error'));
                
                $('#dataTable tbody').html(`
                    <tr>
                        <td colspan="8" class="text-center text-danger py-4">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Error loading data: ${response.error || 'Unknown error'}
                        </td>
                    </tr>
                `);
            }
        },
        error: function(xhr, status, error) {
            console.error("AJAX error details:", {
                status: xhr.status,
                statusText: xhr.statusText,
                responseText: xhr.responseText,
                error: error
            });
            
            showError('Network error: ' + error);
            
            $('#dataTable tbody').html(`
                <tr>
                    <td colspan="8" class="text-center text-danger py-4">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Network error: ${error}
                        <br>
                        <small>Check server logs for details</small>
                    </td>
                </tr>
            `);
        }
    });
}

function handleNoData(filters) {
    console.log("No data found with filters:", filters);
    
    let suggestions = [];
    
    // Generate suggestions based on filters
    if (filters.min_yield && parseFloat(filters.min_yield) < 1000) {
        suggestions.push("Increase minimum yield (try 1000+ kg/ha)");
    }
    
    if (filters.max_yield && parseFloat(filters.max_yield) < 2000) {
        suggestions.push("Increase maximum yield (try 2000+ kg/ha)");
    }
    
    if (Object.keys(filters).filter(k => filters[k] && filters[k] !== '').length >= 3) {
        suggestions.push("Remove one filter at a time");
    }
    
    // Show suggestions
    if (suggestions.length > 0) {
        showInfo(`No data found. Suggestions: ${suggestions.join('; ')}`);
    } else {
        showInfo('No data found with current filters. Try different criteria.');
    }
}

function updateDataTable(data) {
    const tbody = $('#dataTable tbody');
    
    if (!data || data.length === 0) {
        tbody.html(`
            <tr>
                <td colspan="8" class="text-center py-5">
                    <div class="text-warning mb-3">
                        <i class="fas fa-exclamation-triangle fa-3x"></i>
                    </div>
                    <h5 class="text-warning">No Data Found</h5>
                    <p class="text-muted mb-3">Your filters returned 0 results.</p>
                    <div class="d-flex justify-content-center gap-2">
                        <button class="btn btn-success" onclick="resetFilters()">
                            <i class="fas fa-redo me-1"></i> Reset All Filters
                        </button>
                        <button class="btn btn-outline-primary" onclick="suggestRealisticFilters()">
                            <i class="fas fa-lightbulb me-1"></i> Get Suggestions
                        </button>
                    </div>
                    <small class="text-muted mt-3 d-block">
                        <i class="fas fa-info-circle me-1"></i>
                        Tip: Most crops yield ${dataStats.min ? Math.round(dataStats.min) : '1000'} to ${dataStats.max ? Math.round(dataStats.max) : '8000'} kg/ha
                    </small>
                </td>
            </tr>
        `);
        return;
    }
    
    let html = '';
    data.forEach(function(item, index) {
        html += `
            <tr>
                <td>${item.farm_id || index + 1}</td>
                <td>${item.region || 'N/A'}</td>
                <td>${item.crop_type || 'N/A'}</td>
                <td class="fw-bold ${getYieldClass(item.yield_kg_per_hectare)}">
                    ${formatNumber(item.yield_kg_per_hectare)}
                </td>
                <td>${formatNumber(item.soil_moisture_)}</td>
                <td>${formatNumber(item.temperature_C)}</td>
                <td>${formatNumber(item.rainfall_mm)}</td>
                <td>
                    <span class="badge ${getDiseaseClass(item.crop_disease_status)}">
                        ${item.crop_disease_status || 'Unknown'}
                    </span>
                </td>
            </tr>
        `;
    });
    
    tbody.html(html);
    console.log(`✓ Updated table with ${data.length} rows`);
}

function suggestRealisticFilters() {
    console.log("Generating realistic filter suggestions...");
    
    // Clear problematic filters
    if (currentFilters.min_yield && parseFloat(currentFilters.min_yield) < 1000) {
        $('#minYield').val('');
        showInfo('Cleared low minimum yield');
    }
    
    if (currentFilters.max_yield && parseFloat(currentFilters.max_yield) < 2000) {
        $('#maxYield').val('');
        showInfo('Cleared low maximum yield');
    }
    
    // If still no filters, suggest some reasonable defaults
    if (Object.keys(currentFilters).filter(k => currentFilters[k] && currentFilters[k] !== '').length === 0) {
        showInfo('Try applying one filter at a time, starting with Region or Crop Type');
    }
    
    // Re-apply filters with suggestions
    setTimeout(() => {
        applyFilters();
    }, 1000);
}

// Debug/test functions
function testFilters() {
    console.log("Testing with no filters...");
    resetFilters();
}

function testWithRegion() {
    console.log("Testing with region filter...");
    $('#regionFilter').val('North');
    applyFilters();
}

function testAPIEndpoints() {
    console.log("Testing API endpoints...");
    
    // Test each endpoint
    const endpoints = [
        '/api/data/filters',
        '/api/data/stats',
        '/api/charts/yield-distribution'
    ];
    
    endpoints.forEach(endpoint => {
        $.ajax({
            url: endpoint,
            type: 'GET',
            success: function(response) {
                console.log(`✓ ${endpoint}:`, response.success ? 'Success' : 'Failed');
            },
            error: function(error) {
                console.error(`✗ ${endpoint}:`, error);
            }
        });
    });
}

function updateSummaryStats(response) {
    // Update total records
    const totalRecords = response.total_records || 0;
    $('#totalRecords').text(totalRecords);
    
    // Update average yield
    const avgYield = response.average_yield || 0;
    $('#avgYield').text(formatNumber(avgYield) + ' kg/ha');
    
    // Update filter stats card
    $('#filterStats .card-body').html(`
        <div class="row">
            <div class="col-6">
                <small class="text-muted">Total Farms:</small>
                <div class="h5 ${totalRecords === 0 ? 'text-warning' : 'text-success'}">
                    ${totalRecords}
                </div>
            </div>
            <div class="col-6">
                <small class="text-muted">Avg Yield:</small>
                <div class="h5 ${avgYield > 0 ? 'text-success' : 'text-warning'}">
                    ${formatNumber(avgYield)} kg/ha
                </div>
            </div>
        </div>
        ${totalRecords === 0 ? `
        <div class="row mt-2">
            <div class="col-12">
                <small class="text-danger">
                    <i class="fas fa-exclamation-circle me-1"></i>
                    No data with current filters
                </small>
            </div>
        </div>
        ` : ''}
    `);
}

function loadInitialCharts() {
    console.log("Loading initial charts...");
    
    // Wait a bit for charts.js to load
    setTimeout(function() {
        if (typeof window.chartFunctions !== 'undefined') {
            console.log("Loading charts via chartFunctions");
            window.chartFunctions.loadYieldDistribution();
            window.chartFunctions.loadCropPerformance();
            window.chartFunctions.loadRegionAnalysis();
            
            // Load filtered charts after a delay
            setTimeout(function() {
                window.chartFunctions.loadDiseaseImpact();
                window.chartFunctions.loadClimateImpact();
            }, 1000);
        } else {
            console.warn("chartFunctions not available, retrying...");
            setTimeout(loadInitialCharts, 1000);
        }
    }, 500);
}

function updateFilteredCharts() {
    console.log("Updating filtered charts...");
    
    // Update window.currentFilters for charts.js to use
    window.currentFilters = currentFilters;
    
    // Update charts that use filters
    if (typeof window.chartFunctions !== 'undefined') {
        setTimeout(function() {
            window.chartFunctions.loadDiseaseImpact();
            window.chartFunctions.loadClimateImpact();
        }, 500);
    }
}

function exportData() {
    if (Object.keys(currentFilters).length === 0) {
        if (!confirm("No filters applied. Export all data?")) {
            return;
        }
    }
    
    const region = currentFilters.region || '';
    const cropType = currentFilters.crop_type || '';
    
    const url = `/api/export/csv?region=${encodeURIComponent(region)}&crop_type=${encodeURIComponent(cropType)}`;
    window.open(url, '_blank');
}

// Helper functions
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

function showInfo(message) {
    // Remove existing info alerts
    $('.alert-info').remove();
    
    // Show info alert
    $('.container-fluid').prepend(`
        <div class="alert alert-info alert-dismissible fade show" role="alert">
            <i class="fas fa-info-circle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        $('.alert-info').alert('close');
    }, 5000);
}

// Quick filter functions (optional)
function applyQuickFilter(filterType) {
    console.log("Applying quick filter:", filterType);
    
    switch(filterType) {
        case 'high_yield':
            $('#minYield').val('5000');
            $('#maxYield').val('');
            break;
            
        case 'low_yield':
            $('#minYield').val('');
            $('#maxYield').val('2000');
            break;
            
        case 'optimal':
            $('#minYield').val('3000');
            $('#maxYield').val('7000');
            break;
            
        default:
            return;
    }
    
    applyFilters();
}

// Initialize dashboard when page loads
$(document).ready(function() {
    console.log("Dashboard page loaded");
    initializeDashboard();
});