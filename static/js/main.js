// Main JavaScript for Federated Learning Platform

// Global variables
let currentUser = null;
let trainingSession = null;
let charts = {};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkAuthStatus();
});

// Initialize application
function initializeApp() {
    console.log('Federated Learning Platform initialized');
    
    // Add fade-in animation to main content
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.classList.add('fade-in');
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Setup global event listeners
function setupEventListeners() {
    // Handle file drag and drop
    setupDragAndDrop();
    
    // Handle form validations
    setupFormValidations();
    
    // Handle navigation
    setupNavigation();
    
    // Handle real-time updates
    setupRealTimeUpdates();
}

// Check authentication status
function checkAuthStatus() {
    // This would typically check with the server
    // For now, check if we're on a protected page
    const protectedPages = ['/dashboard', '/upload', '/predict'];
    const currentPath = window.location.pathname;
    
    if (protectedPages.some(page => currentPath.includes(page))) {
        // Check if user is logged in (simplified check)
        const hospitalName = document.querySelector('.navbar-nav .dropdown-toggle');
        if (!hospitalName) {
            // Redirect to login if not authenticated
            window.location.href = '/login';
        }
    }
}

// Drag and drop functionality
function setupDragAndDrop() {
    const dropZones = document.querySelectorAll('.upload-area, input[type="file"]');
    
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', handleDragOver);
        zone.addEventListener('dragleave', handleDragLeave);
        zone.addEventListener('drop', handleDrop);
    });
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput) {
            fileInput.files = files;
            // Trigger change event
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    }
}

// Form validation setup
function setupFormValidations() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

// Navigation setup
function setupNavigation() {
    // Add active class to current page
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// Real-time updates setup
function setupRealTimeUpdates() {
    // Poll for training updates if on dashboard
    if (window.location.pathname.includes('/dashboard')) {
        startTrainingStatusPolling();
    }
}

// Training status polling
function startTrainingStatusPolling() {
    setInterval(async () => {
        try {
            const response = await fetch('/api/training_status/current');
            if (response.ok) {
                const status = await response.json();
                updateTrainingStatus(status);
            }
        } catch (error) {
            console.log('No active training session');
        }
    }, 5000); // Poll every 5 seconds
}

function updateTrainingStatus(status) {
    const statusElement = document.getElementById('trainingStatus');
    if (statusElement) {
        statusElement.textContent = status.status || 'Ready';
        
        // Update status indicator class
        statusElement.className = 'status-indicator';
        if (status.status === 'training') {
            statusElement.classList.add('training');
        } else if (status.status === 'completed') {
            statusElement.classList.add('completed');
        } else if (status.status === 'error') {
            statusElement.classList.add('error');
        }
    }
}

// Utility functions
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

// API helper functions
async function apiCall(endpoint, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(endpoint, mergedOptions);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'API call failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        showNotification(`Error: ${error.message}`, 'danger');
        throw error;
    }
}

// Security functions
function sanitizeInput(input) {
    const div = document.createElement('div');
    div.textContent = input;
    return div.innerHTML;
}

function validateCSV(file) {
    return new Promise((resolve, reject) => {
        if (!file) {
            reject(new Error('No file provided'));
            return;
        }
        
        if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
            reject(new Error('File must be a CSV'));
            return;
        }
        
        if (file.size > 50 * 1024 * 1024) { // 50MB
            reject(new Error('File too large (max 50MB)'));
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const csv = e.target.result;
                const lines = csv.split('\n');
                
                if (lines.length < 11) { // Header + at least 10 data rows
                    reject(new Error('CSV must have at least 10 data rows'));
                    return;
                }
                
                const headers = lines[0].split(',');
                if (headers.length < 2) {
                    reject(new Error('CSV must have at least 2 columns'));
                    return;
                }
                
                // Check for suspicious content
                const suspiciousPatterns = [/<script/i, /javascript:/i, /on\w+=/i];
                const content = csv.toLowerCase();
                
                for (const pattern of suspiciousPatterns) {
                    if (pattern.test(content)) {
                        reject(new Error('Suspicious content detected'));
                        return;
                    }
                }
                
                resolve({
                    valid: true,
                    rows: lines.length - 1,
                    columns: headers.length,
                    headers: headers.map(h => h.trim())
                });
                
            } catch (error) {
                reject(new Error('Invalid CSV format'));
            }
        };
        
        reader.onerror = () => reject(new Error('Error reading file'));
        reader.readAsText(file);
    });
}

// Chart utilities
function createChart(canvasId, type, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            }
        }
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: type,
        data: data,
        options: mergedOptions
    });
    
    return charts[canvasId];
}

function updateChart(canvasId, newData) {
    const chart = charts[canvasId];
    if (chart) {
        chart.data = newData;
        chart.update();
    }
}

// Privacy and security indicators
function showPrivacyIndicator(element, level = 'high') {
    const indicator = document.createElement('span');
    indicator.className = 'privacy-indicator ms-2';
    
    const icons = {
        high: 'fas fa-shield-alt',
        medium: 'fas fa-shield-halved',
        low: 'fas fa-exclamation-triangle'
    };
    
    const colors = {
        high: 'success',
        medium: 'warning',
        low: 'danger'
    };
    
    indicator.innerHTML = `<i class="${icons[level]}"></i> ${level.toUpperCase()} PRIVACY`;
    indicator.classList.add(`bg-${colors[level]}`);
    
    element.appendChild(indicator);
}

// Rate limiting handler
let requestCounts = {};

function checkRateLimit(endpoint, limit = 10, window = 3600000) { // 1 hour window
    const now = Date.now();
    const key = endpoint;
    
    if (!requestCounts[key]) {
        requestCounts[key] = [];
    }
    
    // Remove old requests outside the window
    requestCounts[key] = requestCounts[key].filter(time => now - time < window);
    
    if (requestCounts[key].length >= limit) {
        showNotification('Rate limit exceeded. Please try again later.', 'warning');
        return false;
    }
    
    requestCounts[key].push(now);
    return true;
}

// Export functions for global use
window.FedLearn = {
    showNotification,
    formatFileSize,
    formatDate,
    apiCall,
    sanitizeInput,
    validateCSV,
    createChart,
    updateChart,
    showPrivacyIndicator,
    checkRateLimit
};

// Error handling
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showNotification('An unexpected error occurred. Please refresh the page.', 'danger');
});

// Unhandled promise rejection
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showNotification('A network error occurred. Please check your connection.', 'warning');
});

// Page visibility change (pause polling when tab is hidden)
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('Page hidden - pausing updates');
    } else {
        console.log('Page visible - resuming updates');
    }
});

console.log('Federated Learning Platform JavaScript loaded successfully');
