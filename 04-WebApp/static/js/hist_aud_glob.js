// set the dimensions and margins of the graph
var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 600 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// set the ranges
var x_other = d3.scaleBand()
          .range([0, width])
          .padding(0.1);
var y_other = d3.scaleLinear()
          .range([height, 0]);

// append the svg object to the body of the page
var svg_other = d3.select("#audio_emotions_dist_other")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

// get the data
d3.csv("static/js/db/audio_emotions_dist_other.txt", function(error, data) {

  if (error) throw error;

  // format the data
  data.forEach(function(d) {
    d.VALUE = +d.VALUE;
  });

  // Scale the range of the data in the domains
  x_other.domain(data.map(function(d) { return d.EMOTION; }));
  y_other.domain([0, d3.max(data, function(d) { return d.VALUE; })]);

  // append the rectangles for the bar chart
  svg_other.selectAll(".bar")
      .data(data)
    .enter().append("rect")
      .attr("class", "bar")
      .attr("x", function(d) { return x_other(d.EMOTION); })
      .attr("width", x_other.bandwidth())
      .attr("y", function(d) { return y_other(d.VALUE); })
      .attr("height", function(d) { return height - y_other(d.VALUE); })
      .style("fill", "#69b3a2");

  // add the x Axis
  svg_other.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x_other));

  // add the y Axis
  svg_other.append("g")
      .call(d3.axisLeft(y_other));

});
