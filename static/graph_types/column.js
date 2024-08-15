function genColumnGraph(data) {
  return {
    chart: {
      height: 334,
      width: 500,
      type: 'bar'
    },
    dataLabels: {
      enabled: false
    },
    fill: {
      opacity: 1
    },
    plotOptions: {
      bar: {
        horizontal: false,
        columnWidth: '75%',
        endingShape: 'rounded'
      }
    },
    series: data.series.map(serie => ({ name: serie.title, data: serie.data })),
    stroke: {
      show: true,
      width: 2,
      colors: ['transparent']
    },
    title: {
      text: data.title,
      align: 'center'
    },
    xaxis: {
      categories: data.categories,
    },
    yaxis: {
      labels: {
        formatter: val => val?.toPrecision(data.precision) ?? val
      },
      title: {
        text: data.yaxis
      }
    }
  }
}