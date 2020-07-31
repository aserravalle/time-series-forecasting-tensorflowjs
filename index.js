// Machine Learning parameters
let training_periods = 20;
let n_epochs = 100;
let learningrate = 0.07;
let n_hiddenlayers = 20;
let prediction_period = 30;

// Data Parameters
let grouped_data; // array of dictionaries with keys { timestamp, price }
let timestamps; // Nx1 array of dates
let prices; // Nx1 array of prices
let Train;
let X_train; // Last 20 prices in each row
let Y_train; // Average

let DEBUG = true; // if true, will skip the button pressing

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
         timestamps = data.map(function (val) { return val['timestamp']; });
         prices = data.map(function (val) { return val['price']; });

         let graph_plot = document.getElementById('div_linegraph_data');
         Plotly.newPlot(graph_plot, [{
            x: timestamps, y: prices,
            name: "Shipment Revenue",
            xaxis: {title: headers[0]},
            yaxis: {title: headers[1]}
         }], { margin: { t: 0 } });
      }
   }
   displayTrainingData(data);
}

function displayTrainingData(data) {
   $("#div_container_trainingdata").show();

   Train = GenerateTrainingData(data, training_periods);
   /* Train = [{
         target: 12.34, 
         set: [
               {timestamp: "date1", price: 12.34},
               {timestamp: "date2", price: 12.34},
               {timestamp: "date3", price: 12.34}
               ] // 20 rows of grouped_data in each row
         }] */

   let set = Train.map(function (val) { return val['set']; });
   let data_output = "";
   for (let index = 0; index < 10; index++) {
      data_output += "<tr><td width=\"20px\">" + (index + 1) +
         "</td><td>[" + set[index].map(function (val) {
            return (Math.round(val['price'] * 10000) / 10000).toString();
         }).toString() +
         "]</td><td>" + Train[index]['target'] + "</td></tr>";
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

	if (DEBUG) {
      onClickTrainModel();
	}
}

function GenerateTrainingData(data, training_periods) {
   let train = []
   for (let i = 0; i <= data.length - training_periods - 1; i++) {
      train.push({ set: data.slice(i, i + training_periods), target: data[i + training_periods] });
   }
   return train;
}

//================
// Train
//================

async function onClickTrainModel(){
   let epoch_loss = [];

   $("#btn_draw_trainmodel").hide();
   $("#div_container_training").show();

   document.getElementById("div_traininglog").innerHTML = "";

   X_train = Train.map(function (val) {
      return val['set'].map(function (val_set) { return val_set['price']; })
   });
   Y_train = Train.map(function (val) {
      return val['target']
   }).map(function (val2) { return val2['price'] });

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

   result = await trainModel(X_train, Y_train, training_periods, n_epochs, learningrate, n_hiddenlayers, trainingLogCallback);

   let logHtml = document.getElementById("div_traininglog").innerHTML;
   logHtml = "<div>Model training complete</div>" + logHtml;
   document.getElementById("div_traininglog").innerHTML = logHtml;

   $("#div_container_validate").show();
   $("#div_container_validatefirst").hide();
   $("#div_container_predict").show();
   $("#div_container_predictfirst").hide();

   if (DEBUG) {
      onClickValidate();
   }
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
   // validation
   let x_train = X_train.slice(0, X_train.length - prediction_period);
   let y_pred = makePredictions(x_train, result['model']);

   // Timestamps
   let timestamps_train = GetTimestamps(0, grouped_data.length - prediction_period);
   let timestamps_val = GetTimestamps(grouped_data.length - prediction_period, grouped_data.length);
   let timestamps_pred = GetTimestamps(training_periods - 1, grouped_data.length);

   // prices
   let prices_train = prices.slice(0, grouped_data.length - prediction_period);
   let prices_val = prices.slice(grouped_data.length - prediction_period, grouped_data.length);

   // Plotly
   let graph_plot = document.getElementById('div_validation_graph');
   Plotly.newPlot(graph_plot, [{
      x: timestamps_train, y: prices_train, name: "Shipment Revenue", mode: 'lines', line: { color: 'rgb(55, 128, 191)', width: 3}
   }], { margin: { t: 0 } });
   Plotly.plot(graph_plot, [{
      x: timestamps_val, y: prices_val, name: "Shipment Revenue (unseen)", mode: 'lines', line: { color: 'rgb(55, 128, 191)', dash: 'dash', width: 3 }
   }], { margin: { t: 0 } });
   Plotly.plot(graph_plot, [{
      x: timestamps_pred, y: y_pred, name: "Predicted", mode: 'lines', line: { color: 'rgb(165,0,38)', width: 1 }
   }], { margin: { t: 0 } });
}

function GetTimestamps(a, b) {
   return grouped_data.map(function (val) {
      return val['timestamp'];
   }).slice(a, b);
}

//================
// Modelling
//================

async function trainModel(X, Y, training_periods, n_epochs, learning_rate, n_layers, callback) {
   // Define model parameters
   const input_layer_shape = training_periods;
   const input_layer_neurons = 100;
   const rnn_input_layer_features = 10;
   const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;
   const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps];
   const rnn_output_neurons = 20;
   const rnn_batch_size = training_periods;
   const output_layer_shape = rnn_output_neurons;
   const output_layer_neurons = 1;

   // Define data size
   const xs = tf.tensor2d(X, [X.length, X[0].length]);
   const ys = tf.tensor2d(Y, [Y.length, 1]).reshape([Y.length, 1]);

   // Build model layers
   const model = tf.sequential();
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

   // compile model
   model.compile({
      optimizer: tf.train.adam(learning_rate),
      loss: 'meanSquaredError'
   });

   // graph the error reduction
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
   y_preds = []
   N = X.length;
   for (var i = 0; i < N; i++) {
      let y_pred = model.predict(tf.tensor2d(X[i], [1, X[0].length]));
      y_pred = Array.from(y_pred.dataSync())[0];
      y_preds.push(y_pred);
   }

   for (var i = N; i < N + prediction_period; i++) { // we wish to predict (prediction_period) days in the future
      let new_Xi = X[X.length-1]; // Take the last observation

      new_Xi.shift(); // remove the oldest value (index = 0) in the last X array
      new_Xi.push(y_preds[y_preds.length-1]); // Add the last prediction to that array in the newest position (index = length)

      let y_pred = model.predict(tf.tensor2d(new_Xi, [1, X[0].length]));
      y_pred = Array.from(y_pred.dataSync())[0];
      y_preds.push(y_pred); // Make a prediction on the fake array
   }

   return y_preds;
}