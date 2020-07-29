// Machine Learning parameters
let training_periods = 20;
let trainingsize = 90;
let n_epochs = 5;
let learningrate = 0.01;
let n_hiddenlayers = 4;
let add_days = 1;

// Data Parameters
let grouped_data; // array of dictionaries with keys { timestamp, price }
let X_train; // Last 20 data points
let Y_train; // Average

let DEBUG = true; // if true, will skip the whol

$(document).ready(function(){
   $('select').formSelect();
});

//================
// Load Data
//================

function onClickLoadData() {
   $("#btn_train_loading").show();

   let file = document.getElementById("prediction_data_upload").files[0];

	try {
      Papa.parse(file, {
         complete: function (results) {
            CleanRawData(results.data);
         }
      });
   } catch (e) {
      console.log(e)
      $("#btn_train_loading").hide();
      $("#div_trainingdata").html("<span>Error with the uploaded SPR</span>");
	}
}

function CleanRawData(raw_data) {
   /*Current Format
      raw_data = [
      [ "date2", "6.17" ],
      [ "date2", "6.17" ],
      [ "date1", "12.34" ],
      [ "date3", "12.34" ]
      ]
    * Desired Format
      grouped_data = {
         { timestamp: "date1", price: 12.34 },
         { timestamp: "date2", price: 12.34 },
         { timestamp: "date3", price: 12.34 }
      }
   */

   // Remove headers
   let headers = raw_data[0];
   raw_data.shift();

   // Sort by date
   raw_data.sort(function (a, b) {
      if (a[0] == b[0])
         return 0;
      if (a[0] < b[0])
         return -1;
      if (a[0] > b[0])
         return 1;
   });

   grouped_data = GroupRawData(raw_data);
   $("#btn_train_loading").hide();

   GraphCleanData(grouped_data, headers);
}

function GroupRawData(data) {
   let grouped_data = []
   let Timestamp = "";
   let Price = 0;

   data.forEach(function (row, index) { // Currently train.csv has only two columns. Will have to adjust indices so they refer to the columns in the whole report
      let dte = row[0];
      let prc = parseFloat(row[1]) || 0;

      if (Timestamp != row[0]) {
         if (index > 0) {
            grouped_data.push({ timestamp: Timestamp, price: Price });
         }
         Timestamp = dte;
         Price = prc;
      }
      else {
         Price += prc;
      }
      if (index == data.length - 1) {
         grouped_data.push({ timestamp: Timestamp, price: Price });
      }
   });

   return grouped_data;
}

function GraphCleanData(data, headers) {
   $("#div_container_linegraph").show();

   if (data){
      let title = "Shipment Revenue Over Time";
      $("#div_linegraph_data_title").text(title);

      if(data.length > 0){
         let timestamps = data.map(function (val) { return val['timestamp']; });
         let revenues = data.map(function (val) { return val['price']; });

         let graph_plot = document.getElementById('div_linegraph_data');
         Plotly.newPlot(graph_plot, [{
            x: timestamps, y: revenues,
            name: "Shipment Revenue",
            xaxis: {title: headers[0]},
            yaxis: {title: headers[1]}
         }], { margin: { t: 0 } });
      }
   }
   displayTrainingData(data);
}

//================
// Train
//================

function displayTrainingData(data) {
   $("#div_container_trainingdata").show();

   sma_vec = GenerateTrainingData(data, training_periods);
   /* sma_vec = [{
         avg: 12.34, 
         set: [
               {timestamp: "date1", price: 12.34},
               {timestamp: "date2", price: 12.34},
               {timestamp: "date3", price: 12.34}
               ] // 20 rows of grouped_data in each row
         }] */

   let set = sma_vec.map(function (val_sma) { return val_sma['set']; });
   let data_output = "";
   for (let index = 0; index < 10; index++) {
      data_output += "<tr><td width=\"20px\">" + (index + 1) +
         "</td><td>[" + set[index].map(function (val) {
            return (Math.round(val['price'] * 10000) / 10000).toString();
         }).toString() +
         "]</td><td>" + sma_vec[index]['avg'] + "</td></tr>";
   }

   data_output = "<table class='striped'>" +
   "<thead><tr><th scope='col'>#</th>" +
   "<th scope='col'>Input (X)</th>" +
   "<th scope='col'>Label (Y)</th></thead>" +
   "<tbody>" + data_output + "</tbody>" +
   "</table>";

   $("#div_trainingdata").html(data_output);

   $("#div_container_train").show();
   $("#div_container_trainfirst").hide();
}
function GenerateTrainingData(data, training_periods) {
   let r_avgs = []
   let avg_prev = 0;
   for (let i = 0; i <= data.length - training_periods; i++) {
      let curr_avg = 0.00, t = i + training_periods;
      for (let k = i; k < t && k <= data.length; k++) {
         curr_avg += data[k]['price'] / training_periods;
      }
      r_avgs.push({ set: data.slice(i, i + training_periods), avg: curr_avg });
      avg_prev = curr_avg;
   }
   return r_avgs;
}


async function onClickTrainModel(){
   let epoch_loss = [];

   $("#btn_draw_trainmodel").hide();
   $("#div_container_training").show();

   document.getElementById("div_traininglog").innerHTML = "";

   X_train = sma_vec.map(function(val_sma){
      return val_sma['set'].map(function (val_set) { return val_set['price']; })
   });
   Y_train = grouped_data.map(function (val) { return val['price']; }).slice(training_periods - 1, grouped_data.length);

   let x_train = X_train.slice(0, Math.floor(trainingsize / 100 * X_train.length));
   let y_train = Y_train.slice(0, Math.floor(trainingsize / 100 * Y_train.length));

   let trainingLogCallback = function(epoch, log) {
      let logHtml = document.getElementById("div_traininglog").innerHTML;
      logHtml = "<div>Epoch: " + (epoch + 1) + " (of " + n_epochs + ")" +
         ", loss: " + log.loss +
         ", difference: " + (epoch_loss[epoch_loss.length - 1] - log.loss) +
         "</div>" + logHtml;

      epoch_loss.push(log.loss);

      document.getElementById("div_traininglog").innerHTML = logHtml;
      document.getElementById("div_training_progressbar").style.width = Math.ceil(((epoch + 1) * (100 / n_epochs))).toString() + "%";
      document.getElementById("div_training_progressbar").innerHTML = Math.ceil(((epoch + 1) * (100 / n_epochs))).toString() + "%";

      let graph_plot = document.getElementById('div_linegraph_trainloss');
      Plotly.newPlot( graph_plot, [{x: Array.from({length: epoch_loss.length}, (v, k) => k+1), y: epoch_loss, name: "Loss" }], { margin: { t: 0 } } );
   };

   result = await trainModel(x_train, y_train, training_periods, n_epochs, learningrate, n_hiddenlayers, trainingLogCallback);

   let logHtml = document.getElementById("div_traininglog").innerHTML;
   logHtml = "<div>Model training complete</div>" + logHtml;
   document.getElementById("div_traininglog").innerHTML = logHtml;

   $("#div_container_validate").show();
   $("#div_container_validatefirst").hide();
   $("#div_container_predict").show();
   $("#div_container_predictfirst").hide();
}

//================
// Validate
//================

function onClickValidate() {
   $("#div_container_validating").show();
   $("#load_validating").show();
   $("#btn_validation").hide();

   try {
      ValidateAndGraphModel()
   } catch (e) {
      console.log(e)
      $("#btn_validation").show();
   }

$("#load_validating").hide();
}

function ValidateAndGraphModel() {
   // validate on training
   let x_train = X_train.slice(0, Math.floor(trainingsize / 100 * X_train.length));
   let y_pred_train = makePredictions(x_train, result['model']);

   // validate on unseen
   let x_val = X_train.slice(Math.floor(trainingsize / 100 * X_train.length), X_train.length);
   let y_pred_val = makePredictions(x_val, result['model']);

   // Timestamps
   let timestamps_training = GetTimestamps(0, grouped_data.length - Math.floor((100 - trainingsize) / 100 * grouped_data.length) +1 );
   let timestamps_pred_train = GetTimestamps(training_periods - 1, grouped_data.length - Math.floor((100 - trainingsize) / 100 * grouped_data.length) - 1);
   let timestamps_pred_val = GetTimestamps(training_periods - 1 + Math.floor(trainingsize / 100 * X_train.length), X_train.length - 1);

   // prices
   let prices = grouped_data.map(function (val) { return val['price']; });

   // Plotly
   let graph_plot = document.getElementById('div_validation_graph');
   Plotly.newPlot(graph_plot, [{
      x: timestamps_training, y: prices, name: "Shipment Revenue", mode: 'lines', line: { color: 'rgb(55, 128, 191)', width: 3}
   }], { margin: { t: 0 } });
   Plotly.plot(graph_plot, [{
      x: timestamps_pred_val, y: prices, name: "Shipment Revenue (unseen)", mode: 'lines', line: { color: 'rgb(55, 128, 191)', dash: 'dash', width: 3 }
   }], { margin: { t: 0 } });
   Plotly.plot(graph_plot, [{
      x: timestamps_pred_train, y: y_pred_train, name: "Predicted (train)", mode: 'lines', line: { color: 'rgb(165,0,38)', width: 1 }
   }], { margin: { t: 0 } });
   Plotly.plot(graph_plot, [{
      x: timestamps_pred_val, y: y_pred_val, name: "Predicted (test)", mode: 'lines', line: { color: 'rgb(254,224,144)', width: 1 }
   }], { margin: { t: 0 } });
}

function GetTimestamps(a, b) {
   return grouped_data.map(function (val) {
      return val['timestamp'];
   }).splice(a, b);
}

//================
// Modelling
//================

async function trainModel(X, Y, window_size, n_epochs, learning_rate, n_layers, callback) {

   const input_layer_shape = window_size;
   const input_layer_neurons = 100;

   const rnn_input_layer_features = 10;
   const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;

   const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps];
   const rnn_output_neurons = 20;

   const rnn_batch_size = window_size;

   const output_layer_shape = rnn_output_neurons;
   const output_layer_neurons = 1;

   const model = tf.sequential();

   const xs = tf.tensor2d(X, [X.length, X[0].length]).div(tf.scalar(10));
   const ys = tf.tensor2d(Y, [Y.length, 1]).reshape([Y.length, 1]).div(tf.scalar(10));

   model.add(tf.layers.dense({ units: input_layer_neurons, inputShape: [input_layer_shape] }));
   model.add(tf.layers.reshape({ targetShape: rnn_input_shape }));

   let lstm_cells = [];
   for (let index = 0; index < n_layers; index++) {
      lstm_cells.push(tf.layers.lstmCell({ units: rnn_output_neurons }));
   }

   model.add(tf.layers.rnn({
      cell: lstm_cells,
      inputShape: rnn_input_shape,
      returnSequences: false
   }));

   model.add(tf.layers.dense({ units: output_layer_neurons, inputShape: [output_layer_shape] }));

   model.compile({
      optimizer: tf.train.adam(learning_rate),
      loss: 'meanSquaredError'
   });

   const hist = await model.fit(xs, ys,
      {
         batchSize: rnn_batch_size, epochs: n_epochs, callbacks: {
            onEpochEnd: async (epoch, log) => {
               callback(epoch, log);
            }
         }
      });

   return { model: model, stats: hist };
}

function makePredictions(X, model) {
   let predictedResults = model.predict(tf.tensor2d(X, [X.length, X[0].length]).div(tf.scalar(10))).mul(10);
   let result = Array.from(predictedResults.dataSync());

   let sum = result.reduce((a, b) => a + b, 0);
   let avg = (sum / result.length) || 0;
   console.log(avg);
   let scale = 100000 / avg;

   return result.map(function (x) { return x * scale; });
}

