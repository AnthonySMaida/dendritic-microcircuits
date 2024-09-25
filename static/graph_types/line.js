function genLineGraph(data) {
  return {
    chart: {
      height: 334,
      width: 500,
      type: 'line',
      zoom: {
        allowMouseWheelZoom: false,
        autoScaleYaxis: true,
      }
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
    series: data.series.map(serie => ({ name: serie.title, data: serie.data })),
    stroke: {
      curve: 'smooth'
    },
    title: {
      text: data.title,
      align: 'left'
    },
    xaxis: {
      title: data.xaxis ? {
        text: data.xaxis
      } : {},
      type: 'numeric'
    },
    yaxis: {
      labels: {
        formatter: val => val?.toPrecision(data.precision) ?? val
      },
      title: {
        text: data.yaxis
      }
    },
    ...data.extra
  }
}
