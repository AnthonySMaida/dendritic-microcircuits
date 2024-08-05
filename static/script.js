const container = document.getElementById('container')

function generateImgNodeFromBase64(b64) {
  return Object.assign(document.createElement('img'), { src: `data:image/png;base64,${b64}`})
}

function getData() {
    /**
     *  Fetch tells browser to send a request to the flask backend on the '/data' route.
     *  What calls getData() ?
     */
  fetch('/data').then(r => r.json())
    .then(data => {
      cleanContainer()
      const canvas1 = Object.assign(document.createElement('div'), { id: 'graph1' })
      container.appendChild(canvas1)
      const canvas2 = Object.assign(document.createElement('div'), { id: 'graph2' })
      container.appendChild(canvas2)
      const canvas3 = Object.assign(document.createElement('div'), { id: 'graph3' })
      container.appendChild(canvas3)

      const data1 = data.data1.reduce((prev, curr) => [
        [...prev[0], curr[0]],
        [...prev[1], curr[1]]
      ], [[], []])
      const data2 = data.data2.reduce((prev, curr) => [
        [...prev[0], curr[0]],
        [...prev[1], curr[1]],
        [...prev[2], curr[2]]
      ], [[], [], []])
      const data3 = data.data3.reduce((prev, curr) => [
        [...prev[0], curr[0]],
        [...prev[1], curr[1]],
        [...prev[2], curr[2]],
        [...prev[3], curr[3]],
        [...prev[4], curr[4]]
      ], [[], [], [], [], []])

      const options = {
        chart: {
          height: 334,
          width: 500,
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
          text: 'Layer 1 Apical MPs',
          align: 'center'
        },

      })
      chart1.render()


      const chart2 = new ApexCharts(canvas2, {
        ...options,
        series: [
          { name: 'Apical MP 1', data: data2[0] },
          { name: 'Apical MP 2', data: data2[1] },
          { name: 'Apical MP 3', data: data2[2]}
        ],
        title: {
          text: 'Layer 2 Apical MPs',
          align: 'center'
        },

      })
      chart2.render()


      const chart3 = new ApexCharts(canvas3, {
        ...options,
        series: [
          { name: 'Soma act', data: data3[0] },
          { name: 'Basal Act', data: data3[1] },
          { name: 'Post value', data: data3[2] },
          { name: 'Soma mp', data: data3[3] },
          { name: 'Basal mp', data: data3[4] }
        ],
        title: {
          text: 'Learning Rule PP_FF',
          align: 'center'
        }
      })
      chart3.render()
    })
}

function getGraphs() {
  fetch('/graphs').then(r => r.json())
    .then(data => {
      cleanContainer()
      container.appendChild(generateImgNodeFromBase64(data.data1))
      container.appendChild(generateImgNodeFromBase64(data.data2))
      container.appendChild(generateImgNodeFromBase64(data.data3))
    })
    .catch(error => console.error(error))
}

function cleanContainer() {
  Array.from(document.getElementById('container').children).forEach(n => n.remove())
}
