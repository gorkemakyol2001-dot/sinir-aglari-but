// Göz Hastalığı Karar Destek Sistemi - JavaScript

document.addEventListener('DOMContentLoaded', function () {
    // Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const removeBtn = document.getElementById('removeBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resultsSection = document.getElementById('resultsSection');

    let selectedFile = null;

    // Upload area click
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];
        handleFileSelect(file);
    });

    // Remove button
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Predict button
    predictBtn.addEventListener('click', () => {
        if (selectedFile) {
            makePrediction();
        }
    });

    // Handle file selection
    function handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            alert('Geçersiz dosya formatı! Sadece JPG, JPEG ve PNG desteklenir.');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            alert('Dosya boyutu çok büyük! Maksimum 16MB yüklenebilir.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadArea.style.display = 'none';
            imagePreview.style.display = 'block';
            predictBtn.disabled = false;
            resultsSection.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Reset upload
    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        uploadArea.style.display = 'block';
        imagePreview.style.display = 'none';
        predictBtn.disabled = true;
        resultsSection.style.display = 'none';
    }

    // Make prediction
    function makePrediction() {
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Show loading state
        const btnText = predictBtn.querySelector('.btn-text');
        const btnLoader = predictBtn.querySelector('.btn-loader');
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
        predictBtn.disabled = true;

        // Make API request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Hata: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Tahmin yapılırken bir hata oluştu: ' + error.message);
            })
            .finally(() => {
                // Reset button state
                btnText.style.display = 'inline-block';
                btnLoader.style.display = 'none';
                predictBtn.disabled = false;
            });
    }

    // Display results
    function displayResults(data) {
        const { prediction, top_predictions } = data;

        // Main prediction
        document.getElementById('diseaseName').textContent = prediction.disease;
        document.getElementById('confidenceValue').textContent = prediction.confidence.toFixed(2) + '%';
        document.getElementById('confidenceFill').style.width = prediction.confidence + '%';
        document.getElementById('diseaseDescription').innerHTML =
            `<p>${prediction.description}</p>`;

        // Top 3 predictions
        const predictionsList = document.getElementById('predictionsList');
        predictionsList.innerHTML = '';

        top_predictions.forEach((pred, index) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.style.animationDelay = `${index * 0.1}s`;

            item.innerHTML = `
                <div>
                    <div class="prediction-name">${index + 1}. ${pred.disease}</div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 5px;">
                        ${pred.description}
                    </div>
                </div>
                <div class="prediction-confidence">${pred.confidence.toFixed(2)}%</div>
            `;

            predictionsList.appendChild(item);
        });

        // Show results section with animation
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
