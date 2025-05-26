document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const previewArea = document.getElementById('preview-area');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const detectButton = document.getElementById('detect-button');
    const cancelButton = document.getElementById('cancel-button');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const detectionResults = document.getElementById('detection-results');
    const loadingOverlay = document.getElementById('loading-overlay');
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const downloadImageBtn = document.getElementById('download-image');
    const zoomImageBtn = document.getElementById('zoom-image');
    const zoomModal = document.getElementById('zoom-modal');
    const zoomedImage = document.getElementById('zoomed-image');
    const closeModal = document.querySelector('.close');
    const exportJsonBtn = document.getElementById('export-json');

    // Variables
    let currentFile = null;
    let detectionData = null;

    // Event Listeners
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    detectButton.addEventListener('click', () => {
        if (currentFile) {
            detectObjects(currentFile);
        }
    });

    cancelButton.addEventListener('click', resetUpload);

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));

            // Add active class to clicked button and corresponding pane
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    downloadImageBtn.addEventListener('click', () => {
        if (resultImage.src) {
            const link = document.createElement('a');
            link.href = resultImage.src;
            link.download = 'detection_result.png';
            link.click();
        }
    });

    zoomImageBtn.addEventListener('click', () => {
        if (resultImage.src) {
            zoomedImage.src = resultImage.src;
            zoomModal.style.display = 'block';
        }
    });

    closeModal.addEventListener('click', () => {
        zoomModal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === zoomModal) {
            zoomModal.style.display = 'none';
        }
    });

    exportJsonBtn.addEventListener('click', () => {
        if (detectionData) {
            const dataStr = JSON.stringify(detectionData, null, 2);
            const blob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'detection_data.json';
            link.click();
            URL.revokeObjectURL(url);
        }
    });

    // Functions
    function handleFile(file) {
        // Check if file is an image
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }

        currentFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewArea.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    function detectObjects(file) {
        // Show loading overlay
        loadingOverlay.style.display = 'flex';

        // Create form data
        const formData = new FormData();
        formData.append('image', file);
        // Get selected model name from a dropdown (if exists)
        const modelSelect = document.getElementById('model-select');
        if (modelSelect && modelSelect.value) {
            formData.append('model_name', modelSelect.value);
        }

        fetch('/detect', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || 'Network response was not ok');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Hide loading overlay
                loadingOverlay.style.display = 'none';

                // Store detection data
                detectionData = data;

                // Log the data to console for debugging
                console.log('API Response:', data);

                // Display results
                displayResults(data);
            })
            .catch(error => {
                loadingOverlay.style.display = 'none';
                alert('Error: ' + error.message);
                console.error('Error:', error);
            });
    }

    function displayResults(data) {
        // Show results section
        resultsSection.style.display = 'block';

        // Set result image
        if (data.image_url) {
            resultImage.src = data.image_url;
            console.log('Setting image URL to:', data.image_url);
        } else {
            console.error('No image URL in response');
        }

        // Clear previous results
        detectionResults.innerHTML = '';

        // Add detection results to table
        if (data.boxes && data.boxes.length > 0) {
            data.boxes.forEach(box => {
                const row = document.createElement('tr');

                // Convert normalized coordinates back to pixel values if needed
                const x1 = box.x1 !== undefined ? box.x1 : (box.x_center - box.width / 2);
                const y1 = box.y1 !== undefined ? box.y1 : (box.y_center - box.height / 2);
                const x2 = box.x2 !== undefined ? box.x2 : (box.x_center + box.width / 2);
                const y2 = box.y2 !== undefined ? box.y2 : (box.y_center + box.height / 2);

                const className = typeof box.class === 'string' ? box.class : `Class ${box.class}`;

                row.innerHTML = `
                    <td>${className}</td>
                    <td>${(box.confidence * 100).toFixed(2)}%</td>
                    <td>(${x1.toFixed(2)}, ${y1.toFixed(2)}, ${x2.toFixed(2)}, ${y2.toFixed(2)})</td>
                `;
                detectionResults.appendChild(row);
            });
        } else {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="3">No objects detected</td>';
            detectionResults.appendChild(row);
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function resetUpload() {
        // Reset file input
        fileInput.value = '';
        currentFile = null;

        // Hide preview
        previewArea.style.display = 'none';
        uploadArea.style.display = 'block';

        // Clear preview image
        previewImage.src = '';
    }
});
