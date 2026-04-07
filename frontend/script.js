let importanceChart = null; // Store chart instance globally to destroy it on refresh

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = document.getElementById('submit-btn');
    const resultCard = document.getElementById('result-card');
    const errorMsg = document.getElementById('error-message');
    
    btn.disabled = true;
    btn.textContent = 'Running AI Pipeline...';
    resultCard.classList.add('hidden');
    errorMsg.classList.add('hidden');

    const requestData = {
        buying: document.getElementById('buying').value,
        maint: document.getElementById('maint').value,
        doors: document.getElementById('doors').value,
        persons: document.getElementById('persons').value,
        lug_boot: document.getElementById('lug_boot').value,
        safety: document.getElementById('safety').value
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error);

        // 1. Draw Stars
        document.getElementById('stars-container').innerHTML = 
            '★'.repeat(data.rating) + '☆'.repeat(5 - data.rating);
        
        // 2. Set Explanation Text
        document.getElementById('explanation-text').textContent = data.explanation;

        // 3. Draw Chart.js Graph
        renderChart(data.feature_importance);

        resultCard.classList.remove('hidden');

    } catch (error) {
        errorMsg.textContent = `Error: ${error.message}`;
        errorMsg.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run AI Analysis';
    }
});

function renderChart(importanceData) {
    const ctx = document.getElementById('importanceChart').getContext('2d');
    
    // Destroy old chart if it exists so we don't overlap them
    if (importanceChart) {
        importanceChart.destroy();
    }

    const labels = Object.keys(importanceData);
    const values = Object.values(importanceData);

    importanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Influence Weight (%)',
                data: values,
                backgroundColor: '#3498db',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y', // Makes it a horizontal bar chart
            scales: {
                x: { beginAtZero: true, max: 100 }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}