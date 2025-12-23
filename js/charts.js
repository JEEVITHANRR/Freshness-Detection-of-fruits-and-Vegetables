document.addEventListener("DOMContentLoaded", function () {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            renderFreshnessChart(data);
            renderCategoryChart(data.categories);
        })
        .catch(error => console.error("Error fetching stats:", error));

    function renderFreshnessChart(data) {
        const ctx = document.getElementById('freshnessChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Fresh', 'Near Expiry', 'Expired'],
                datasets: [{
                    data: [data.fresh, data.near_expiry, data.expired],
                    backgroundColor: ['#27ae60', '#f39c12', '#e74c3c']
                }]
            }
        });
    }

    function renderCategoryChart(categories) {
        const ctx = document.getElementById('categoryChart').getContext('2d');
        const labels = Object.keys(categories);
        const values = Object.values(categories);

        // Assign colors based on category
        const categoryColors = labels.map(label => {
            const l = label.toLowerCase();
            if (l.includes('fruit')) return '#FF6B6B'; // Red
            if (l.includes('veg')) return '#2ECC71';  // Green
            if (l.includes('baked') || l.includes('bakery')) return '#F39C12'; // Orange/Brown
            if (l.includes('dairy')) return '#ECF0F1'; // White/Grey
            if (l.includes('beverage') || l.includes('drink')) return '#3498DB'; // Blue
            if (l.includes('meat')) return '#C0392B'; // Dark Red
            if (l.includes('pack')) return '#9B59B6'; // Purple
            return '#95A5A6'; // Grey default
        });

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Items',
                    data: values,
                    backgroundColor: categoryColors,
                    borderColor: '#2C3E50',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
});
