const container = document.getElementById('container')

function cleanContainer() {
  Array.from(container.children).forEach(n => n.remove())
}

function getData() {
  fetch('/data').then(r => r.json())
    .then(json => {
      cleanContainer()

      const options = {
        chart: {
          height: 400,
          width: 600,
          type: 'line'
        },
        dataLabels: {
          enabled: false
        },
        grid: {
          row: {
            colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
            opacity: 0.5
          },
        },
        stroke: {
          curve: 'smooth'
        },
        xaxis: {
          type: 'numeric'
        },
        yaxis: {
          labels: {
            formatter: val => val.toPrecision(2)
          },
          title: {
            text: 'Loss'
          }
        }
      }

      for (const [title, data] of Object.entries(json)) {
        if (!Array.isArray(data)) continue;

        const canvas = document.createElement('div')
        container.appendChild(canvas)

        const chart = new ApexCharts(canvas, {
          ...options,
          series: data.map(serie => ({ name: serie.title, data: serie.data })),
          title: {
            text: title,
            align: 'center'
          }
        })
        chart.render()
      }
    })
}
