const container = document.getElementById('container')

function generateImgNodeFromBase64(b64) {
  return Object.assign(document.createElement('img'), { src: `data:image/png;base64,${b64}`})
}

function getData() {
  fetch('/data').then(r => r.json())
    .then(data => {
      cleanContainer()
      const canvas1 = Object.assign(document.createElement('div'), { id: 'graph1' })
      container.appendChild(canvas1)
      const canvas2 = Object.assign(document.createElement('div'), { id: 'graph2' })
      container.appendChild(canvas2)

      const data1 = data.data1.reduce((prev, curr) => [
        [...prev[0], curr[0]],
        [...prev[1], curr[1]]
      ], [[], []])
      const data2 = data.data2.reduce((prev, curr) => [
        [...prev[0], curr[0]],
        [...prev[1], curr[1]],
        [...prev[2], curr[2]]
      ], [[], [], []])

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

      const chart1 = new ApexCharts(canvas1, {
        ...options,
        series: [
          { name: 'Apical MP 1', data: data1[0] },
          { name: 'Apical MP 2', data: data1[1] }
        ],
        title: {
          text: 'Layer 1',
          align: 'center'
        },

      })
      chart1.render()

      const chart2 = new ApexCharts(canvas2, {
        ...options,
        series: [
          { name: 'Apical MP 1', data: data2[0] },
          { name: 'Apical MP 2', data: data2[1] },
          { name: 'Apical MP 3', data: data2[2] }
        ],
        title: {
          text: 'Layer 2',
          align: 'center'
        }
      })
      chart2.render()
    })
}

function getGraphs() {
  fetch('/graphs').then(r => r.json())
    .then(data => {
      cleanContainer()
      container.appendChild(generateImgNodeFromBase64(data.data1))
      container.appendChild(generateImgNodeFromBase64(data.data2))
    })
    .catch(error => console.error(error))
}

function cleanContainer() {
  Array.from(document.getElementById('container').children).forEach(n => n.remove())
}
