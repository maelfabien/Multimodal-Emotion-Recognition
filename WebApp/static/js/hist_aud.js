// Global const
var margin = {top: 30, right: 30, bottom: 30, left: 30}
const w = 600 - margin.left - margin.right;
const h = 600 - margin.top - margin.bottom;

let dataset = [];

// Create SVG element for histogram
let svg_hist_dens = d3.select("#hist_density")
                .append("svg")
                  .attr("width", w + margin.left + margin.right)
                  .attr("height", h/2 + margin.top + margin.bottom)
                .append("g")
                  .attr("transform","translate(" + 1.3 * margin.left + "," + margin.top + ")")

// Color scale for density
var myColor = d3.scaleSequential().domain([0,5000]).interpolator(d3.interpolateBrBG)

// Draw function for density histogram
function draw_hist_density() {

  // X axis
  var x_hist = d3.scaleLinear()
      .domain([0, d3.max(dataset, function(d) { return d.EMOTIONS; })])
      .range([0, w]);

  // X axis legend
  svg_hist_dens.append("g")
      .attr("transform", "translate(0," + h/2 + ")")
      .call(d3.axisBottom(x_hist));

  // Create histogram
  var histogram = d3.histogram()
       .value((d) => d.EMOTIONS)
       .domain(x_hist.domain())
       .thresholds(x_hist.ticks(100))

  // Create bins
  var bins = histogram(dataset)

  // Y axis
  var y_hist = d3.scaleLinear()
                  .range([h/2, 0])
                  .domain([0, d3.max(bins, (d) => d.length)])

  // Y axis legend
  svg_hist_dens.append("g")
     .attr("class", "y axis")
     .call(d3.axisLeft(y_hist));

  // Plot histogram
  svg_hist_dens.selectAll("rect")
    .data(bins)
    .enter()
    .append("rect")
      .attr("x", 1)
      .attr("transform", function(d) { return "translate(" + x_hist(d.x0) + "," + y_hist(d.length) + ")"; })
      .attr("width", 20)
      .attr("height", function(d) { return h/2 - y_hist(d.length); })
      .attr("data-legend","Other Candidates")
      .style("fill", "#69b3a2")

  svg_hist_dens.selectAll("new_rect")
    .data(bins)
    .enter()
    .append("rect")
      .attr("x", 1)
      .attr("transform", function(d) { return "translate(" + x_hist(d.x0) + "," + y_hist(d.length) + ")"; })
      .attr("width", 10)
      .attr("height", function(d) { return h/2 - y_hist(d.length); })
      .attr("data-legend","You")
      .style("fill", "#ff0000")

  legend = svg_hist_dens.append("g")
    .attr("class","legend")
    .attr("transform","translate(50,30)")
    .style("font-size","12px")
    .call(d3.legend)

};


// Load and cast data
d3.csv("static/js/audio_emotions.txt")
  .row( (d, i) => {
    return {
      EMOTIONS: +d.EMOTIONS
    };
  }
)
  .get( (error, rows) => {
    console.log("Loaded " + rows.length + " rows");
    if (rows.length > 0){
       console.log("First row: ", rows[0])
       console.log("Last row " , rows[rows.length - 1])
    }
    dataset = rows;
    draw_hist_density();
  }
);
