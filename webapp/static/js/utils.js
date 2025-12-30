// Utility functions
function formatNumber(num) {
    if (num === undefined || num === null) return 'N/A';
    return Number(num).toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
    });
}

function showLoading(element) {
    $(element).html(`
        <div class="text-center">
            <div class="spinner-border text-success" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `);
}

function hideLoading(element, originalContent) {
    $(element).html(originalContent);
}