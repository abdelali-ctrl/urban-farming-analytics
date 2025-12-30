// data_explorer.js - Interactive data explorer

let explorerData = [];
let filteredData = [];
let currentPage = 1;
const pageSize = 10;

function initializeDataExplorer() {
    console.log("Initializing data explorer...");
    
    // Load data
    loadExplorerData();
    
    // Setup event listeners
    setupExplorerListeners();
}

function loadExplorerData() {
    console.log("Loading explorer data...");
    
    $.ajax({
        url: '/api/data/analysis',
        type: 'GET',
        success: function(response) {
            if (response.success && response.data) {
                explorerData = response.data;
                filteredData = [...explorerData];
                
                console.log(`Loaded ${explorerData.length} records for exploration`);
                
                // Populate filter dropdowns
                populateFilterDropdowns();
                
                // Load initial data
                updateDataTable();
                updateSummaryStats();
                createInitialChart();
                
            } else {
                console.error("Failed to load explorer data:", response);
                showExplorerError('Failed to load data for exploration');
            }
        },
        error: function(error) {
            console.error("Error loading explorer data:", error);
            showExplorerError('Error loading data: ' + error);
        }
    });
}

function populateFilterDropdowns() {
    // Get unique values for each filter
    const regions = [...new Set(explorerData.map(d => d.region))].sort();
    const crops = [...new Set(explorerData.map(d => d.crop_type))].sort();
    const irrigations = [...new Set(explorerData.map(d => d.irrigation_type))].sort();
    const diseases = [...new Set(explorerData.map(d => d.crop_disease_status))].sort();
    
    // Populate dropdowns
    populateDropdown('#explorerRegion', regions);
    populateDropdown('#explorerCrop', crops);
    populateDropdown('#explorerIrrigation', irrigations);
    populateDropdown('#explorerDisease', diseases);
}

function populateDropdown(selector, options) {
    const dropdown = $(selector);
    dropdown.empty();
    dropdown.append('<option value="">Toutes les options</option>');
    
    if (options && Array.isArray(options)) {
        options.forEach(option => {
            if (option && option.toString().trim() !== '') {
                dropdown.append(`<option value="${option}">${option}</option>`);
            }
        });
    }
}

function setupExplorerListeners() {
    // Filter buttons
    $('#explorerApplyFilters').on('click', applyExplorerFilters);
    $('#explorerReset').on('click', resetExplorerFilters);
    $('#explorerExport').on('click', exportExplorerData);
    
    // Chart controls
    $('#chartType, #chartX, #chartY, #chartColor').on('change', updateChart);
    
    // Search input
    $('#dataSearch').on('input', function() {
        const searchTerm = $(this).val().toLowerCase();
        if (searchTerm) {
            filterTableBySearch(searchTerm);
        } else {
            updateDataTable();
        }
    });
}

function applyExplorerFilters() {
    console.log("Applying explorer filters...");
    
    // Get filter values
    const filters = {
        region: $('#explorerRegion').val(),
        crop_type: $('#explorerCrop').val(),
        irrigation_type: $('#explorerIrrigation').val(),
        crop_disease_status: $('#explorerDisease').val()
    };
    
    // Apply filters
    filteredData = explorerData.filter(item => {
        let match = true;
        
        if (filters.region && item.region !== filters.region) match = false;
        if (filters.crop_type && item.crop_type !== filters.crop_type) match = false;
        if (filters.irrigation_type && item.irrigation_type !== filters.irrigation_type) match = false;
        if (filters.crop_disease_status && item.crop_disease_status !== filters.crop_disease_status) match = false;
        
        return match;
    });
    
    // Reset to first page
    currentPage = 1;
    
    // Update UI
    updateDataTable();
    updateSummaryStats();
    updateChart();
    
    console.log(`Filtered to ${filteredData.length} records`);
}

function resetExplorerFilters() {
    console.log("Resetting explorer filters...");
    
    // Reset filter values
    $('#explorerRegion').val('');
    $('#explorerCrop').val('');
    $('#explorerIrrigation').val('');
    $('#explorerDisease').val('');
    $('#dataSearch').val('');
    
    // Reset data
    filteredData = [...explorerData];
    currentPage = 1;
    
    // Update UI
    updateDataTable();
    updateSummaryStats();
    updateChart();
}

function updateDataTable() {
    const tbody = $('#explorerTable tbody');
    
    if (!filteredData || filteredData.length === 0) {
        tbody.html(`
            <tr>
                <td colspan="9" class="text-center py-5">
                    <div class="text-muted">
                        <i class="fas fa-database fa-3x mb-3"></i>
                        <h5>Aucune donnée trouvée</h5>
                        <p>Essayez d'ajuster vos filtres</p>
                    </div>
                </td>
            </tr>
        `);
        
        updatePagination();
        return;
    }
    
    // Calculate pagination
    const totalPages = Math.ceil(filteredData.length / pageSize);
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, filteredData.length);
    const pageData = filteredData.slice(startIndex, endIndex);
    
    // Build table rows
    let html = '';
    pageData.forEach((item, index) => {
        const rowIndex = startIndex + index + 1;
        
        html += `
            <tr class="data-point" onclick="viewDataDetails(${rowIndex - 1})">
                <td>${item.farm_id || rowIndex}</td>
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
                <td>
                    <button class="btn btn-sm btn-outline-info" onclick="event.stopPropagation(); viewDataDetails(${rowIndex - 1})">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    });
    
    tbody.html(html);
    
    // Update pagination info
    $('#startRecord').text(startIndex + 1);
    $('#endRecord').text(endIndex);
    $('#totalRecords').text(filteredData.length);
    
    updatePagination();
}

function updatePagination() {
    const pagination = $('#explorerPagination');
    const totalPages = Math.ceil(filteredData.length / pageSize);
    
    if (totalPages <= 1) {
        pagination.empty();
        return;
    }
    
    let html = '';
    
    // Previous button
    html += `
        <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage - 1})">
                <i class="fas fa-chevron-left"></i>
            </a>
        </li>
    `;
    
    // Page numbers
    const maxPagesToShow = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxPagesToShow / 2));
    let endPage = Math.min(totalPages, startPage + maxPagesToShow - 1);
    
    if (endPage - startPage + 1 < maxPagesToShow) {
        startPage = Math.max(1, endPage - maxPagesToShow + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
        html += `
            <li class="page-item ${i === currentPage ? 'active' : ''}">
                <a class="page-link" href="#" onclick="changePage(${i})">${i}</a>
            </li>
        `;
    }
    
    // Next button
    html += `
        <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage + 1})">
                <i class="fas fa-chevron-right"></i>
            </a>
        </li>
    `;
    
    pagination.html(html);
}

function changePage(page) {
    if (page < 1 || page > Math.ceil(filteredData.length / pageSize)) return;
    
    currentPage = page;
    updateDataTable();
    
    // Scroll to top of table
    $('html, body').animate({
        scrollTop: $('#explorerTable').offset().top - 100
    }, 300);
}

function updateSummaryStats() {
    if (filteredData.length === 0) {
        $('#explorerTotalRecords').text('0');
        $('#explorerAvgYield').text('0 kg/ha');
        $('#explorerUniqueCrops').text('0');
        $('#explorerUniqueRegions').text('0');
        return;
    }
    
    // Total records
    $('#explorerTotalRecords').text(filteredData.length.toLocaleString());
    
    // Average yield
    const yields = filteredData.map(d => parseFloat(d.yield_kg_per_hectare)).filter(v => !isNaN(v));
    const avgYield = yields.length > 0 ? yields.reduce((a, b) => a + b, 0) / yields.length : 0;
    $('#explorerAvgYield').text(formatNumber(avgYield) + ' kg/ha');
    
    // Unique crops
    const uniqueCrops = [...new Set(filteredData.map(d => d.crop_type))].length;
    $('#explorerUniqueCrops').text(uniqueCrops);
    
    // Unique regions
    const uniqueRegions = [...new Set(filteredData.map(d => d.region))].length;
    $('#explorerUniqueRegions').text(uniqueRegions);
}

function createInitialChart() {
    updateChart();
}

function updateChart() {
    if (filteredData.length === 0) {
        $('#explorerChart').html(`
            <div class="alert alert-warning m-3">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Aucune donnée à afficher
            </div>
        `);
        return;
    }
    
    const chartType = $('#chartType').val();
    const xVar = $('#chartX').val();
    const yVar = $('#chartY').val();
    const colorVar = $('#chartColor').val();
    
    // Filter valid data
    const validData = filteredData.filter(d => 
        d[xVar] !== null && d[yVar] !== null && 
        !isNaN(d[xVar]) && !isNaN(d[yVar])
    );
    
    if (validData.length === 0) {
        $('#explorerChart').html(`
            <div class="alert alert-warning m-3">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Données insuffisantes pour créer le graphique
            </div>
        `);
        return;
    }
    
    // Update statistics
    updateChartStats(validData, xVar, yVar);
    
    // Create chart based on type
    let trace;
    const xValues = validData.map(d => parseFloat(d[xVar]));
    const yValues = validData.map(d => parseFloat(d[yVar]));
    
    switch(chartType) {
        case 'scatter':
            trace = {
                x: xValues,
                y: yValues,
                mode: 'markers',
                type: 'scatter',
                name: 'Données',
                marker: {
                    size: 8,
                    opacity: 0.6,
                    color: colorVar === 'none' ? '#3498db' : validData.map(d => getColorForCategory(d[colorVar]))
                },
                text: validData.map(d => 
                    `Région: ${d.region}<br>Culture: ${d.crop_type}<br>${xVar}: ${d[xVar]}<br>${yVar}: ${d[yVar]}`
                ),
                hoverinfo: 'text'
            };
            break;
            
        case 'bar':
            // Group data for bar chart
            const groupedData = {};
            validData.forEach(d => {
                const group = d[xVar];
                if (!groupedData[group]) {
                    groupedData[group] = [];
                }
                groupedData[group].push(parseFloat(d[yVar]));
            });
            
            const groups = Object.keys(groupedData);
            const averages = groups.map(group => {
                const values = groupedData[group];
                return values.reduce((a, b) => a + b, 0) / values.length;
            });
            
            trace = {
                x: groups,
                y: averages,
                type: 'bar',
                name: 'Moyenne',
                marker: {
                    color: '#2ecc71'
                }
            };
            break;
            
        case 'line':
            // Sort by x values
            const sortedData = [...validData].sort((a, b) => parseFloat(a[xVar]) - parseFloat(b[xVar]));
            trace = {
                x: sortedData.map(d => parseFloat(d[xVar])),
                y: sortedData.map(d => parseFloat(d[yVar])),
                mode: 'lines+markers',
                type: 'scatter',
                name: 'Courbe',
                line: {
                    color: '#e74c3c',
                    width: 2
                }
            };
            break;
            
        case 'histogram':
            trace = {
                x: xValues,
                type: 'histogram',
                name: 'Distribution',
                marker: {
                    color: '#9b59b6'
                }
            };
            break;
            
        case 'box':
            trace = {
                y: yValues,
                type: 'box',
                name: yVar,
                boxpoints: 'outliers',
                marker: {
                    color: '#f39c12'
                }
            };
            break;
    }
    
    const layout = {
        title: `${yVar} vs ${xVar}`,
        xaxis: { title: xVar.replace('_', ' ') },
        yaxis: { title: yVar.replace('_', ' ') },
        height: 400,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        showlegend: chartType !== 'scatter' || colorVar === 'none'
    };
    
    Plotly.newPlot('explorerChart', [trace], layout);
}

function updateChartStats(data, xVar, yVar) {
    if (data.length === 0) return;
    
    // Calculate correlation
    const xValues = data.map(d => parseFloat(d[xVar]));
    const yValues = data.map(d => parseFloat(d[yVar]));
    const correlation = calculateCorrelation(xValues, yValues);
    
    // Calculate statistics
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const xSorted = [...xValues].sort((a, b) => a - b);
    const ySorted = [...yValues].sort((a, b) => a - b);
    const xMedian = xSorted[Math.floor(xSorted.length / 2)];
    const yMedian = ySorted[Math.floor(ySorted.length / 2)];
    
    // Update UI
    $('#correlationInfo').html(`
        <strong>Corrélation:</strong> ${correlation.toFixed(3)}<br>
        <small>${getCorrelationInterpretation(correlation)}</small>
    `);
    
    $('#distributionInfo').html(`
        <strong>Données:</strong> ${data.length} points<br>
        <strong>Plage X:</strong> ${formatNumber(xMin)} - ${formatNumber(xMax)}<br>
        <strong>Plage Y:</strong> ${formatNumber(yMin)} - ${formatNumber(yMax)}
    `);
    
    $('#minValue').text(formatNumber(yMin));
    $('#maxValue').text(formatNumber(yMax));
    $('#medianValue').text(formatNumber(yMedian));
}

function filterTableBySearch(searchTerm) {
    const filtered = filteredData.filter(item => {
        return Object.values(item).some(value => 
            value && value.toString().toLowerCase().includes(searchTerm)
        );
    });
    
    // Temporarily use filtered data for display
    const tempData = filteredData;
    filteredData = filtered;
    updateDataTable();
    filteredData = tempData; // Restore original filtered data
}

function exportExplorerData() {
    // Create CSV content
    let csvContent = "ID,Région,Culture,Rendement (kg/ha),Humidité sol (%),Température (°C),Précipitations (mm),Statut maladie\n";
    
    filteredData.forEach(item => {
        const row = [
            item.farm_id || '',
            item.region || '',
            item.crop_type || '',
            item.yield_kg_per_hectare || '',
            item.soil_moisture_ || '',
            item.temperature_C || '',
            item.rainfall_mm || '',
            item.crop_disease_status || ''
        ].map(val => `"${val}"`).join(',');
        
        csvContent += row + '\n';
    });
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `smart_farming_data_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function viewDataDetails(index) {
    const item = filteredData[index];
    if (!item) return;
    
    // Create modal content
    const modalContent = `
        <div class="modal fade" id="dataDetailModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header bg-success text-white">
                        <h5 class="modal-title">
                            <i class="fas fa-info-circle me-2"></i>
                            Détails de l'enregistrement
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Informations de base</h6>
                                <table class="table table-sm">
                                    <tr><td>ID Ferme:</td><td><strong>${item.farm_id || 'N/A'}</strong></td></tr>
                                    <tr><td>Région:</td><td>${item.region || 'N/A'}</td></tr>
                                    <tr><td>Culture:</td><td>${item.crop_type || 'N/A'}</td></tr>
                                    <tr><td>Type d'irrigation:</td><td>${item.irrigation_type || 'N/A'}</td></tr>
                                    <tr><td>Type d'engrais:</td><td>${item.fertilizer_type || 'N/A'}</td></tr>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <h6>Performances</h6>
                                <table class="table table-sm">
                                    <tr><td>Rendement:</td><td class="fw-bold ${getYieldClass(item.yield_kg_per_hectare)}">
                                        ${formatNumber(item.yield_kg_per_hectare)} kg/ha
                                    </td></tr>
                                    <tr><td>Statut maladie:</td><td>
                                        <span class="badge ${getDiseaseClass(item.crop_disease_status)}">
                                            ${item.crop_disease_status || 'Unknown'}
                                        </span>
                                    </td></tr>
                                </table>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-12">
                                <h6>Conditions environnementales</h6>
                                <div class="row">
                                    <div class="col-md-3">
                                        <div class="card text-center">
                                            <div class="card-body">
                                                <h5>${formatNumber(item.soil_moisture_)}%</h5>
                                                <small class="text-muted">Humidité sol</small>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card text-center">
                                            <div class="card-body">
                                                <h5>${formatNumber(item.temperature_C)}°C</h5>
                                                <small class="text-muted">Température</small>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card text-center">
                                            <div class="card-body">
                                                <h5>${formatNumber(item.rainfall_mm)}mm</h5>
                                                <small class="text-muted">Précipitations</small>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card text-center">
                                            <div class="card-body">
                                                <h5>${formatNumber(item.NDVI_index)}</h5>
                                                <small class="text-muted">Indice NDVI</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Fermer</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal
    $('#dataDetailModal').remove();
    
    // Add new modal
    $('body').append(modalContent);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('dataDetailModal'));
    modal.show();
}

// Helper functions
function formatNumber(num) {
    if (num === undefined || num === null || isNaN(num)) return 'N/A';
    
    const number = parseFloat(num);
    if (Number.isInteger(number)) {
        return number.toLocaleString('fr-FR');
    } else {
        return number.toLocaleString('fr-FR', {
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
    if (status.includes('no disease') || status.includes('healthy') || status.includes('sain')) return 'bg-success';
    if (status.includes('mild') || status.includes('léger')) return 'bg-info';
    if (status.includes('moderate') || status.includes('modéré')) return 'bg-warning';
    if (status.includes('severe') || status.includes('sévère')) return 'bg-danger';
    return 'bg-secondary';
}

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

function getCorrelationInterpretation(correlation) {
    const absCorr = Math.abs(correlation);
    if (absCorr >= 0.8) return 'Corrélation forte';
    if (absCorr >= 0.5) return 'Corrélation modérée';
    if (absCorr >= 0.3) return 'Corrélation faible';
    return 'Pas de corrélation significative';
}

function getColorForCategory(category) {
    // Simple hash function for consistent colors
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

function showExplorerError(message) {
    $('.container-fluid').prepend(`
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
}

// Initialize on page load
$(document).ready(function() {
    console.log("Data explorer page loaded");
    initializeDataExplorer();
});