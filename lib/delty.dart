library delty;
import 'dart:math' as Math;

class Delty {
  num inputLength;
  num outputLength;

  num bias = 0.6;
  num delta = 0.1;

  var weights;

  Delty(this.inputLength, this.outputLength) {

    weights = new List<List<num>>(outputLength);

    for(num i = 0; i < outputLength; i++){
      var tmp = new List<num>();
      for(num i = 0; i < inputLength; i++){
        tmp.add(0.5);
      }
      weights[i] = tmp;
    }
    print(weights);
  }

  num act(num x){
    return 1 / (1 + Math.pow(Math.E, (-1 * x)));
  }

  num out(num x) {
    if(x > bias)
      return 1;
    return 0;
  }

  num networth(List<num> input, num row) {
    assert(input.length == weights[row].length);

    num net = 0.0;
    for(num i = 0; i < input.length; i++){
      net += input[i] * weights[row][i];
    }
    return net;
  }

  void train(List input, List expected) {
    assert(input.length == inputLength);
    assert(expected.length == outputLength);

    // for each outputneuron
    for(num row = 0; row < outputLength; row++){

      // Propagierungsfunktion: aufsummieren von allen (inputs * weights)
      var net = networth(input, row);

      // Aktivierungsfunktion
      var a = act(net);

      // Ausgabefunktion
      var o = a; //out(a);

      var error = (expected[row] - o);
      var deltaError = delta * error;

      for(num i = 0; i < input.length; i++){
        weights[row][i] += deltaError * input[i];
      }
      print("training: weights = $weights");
    }
  }

  List use(List input) {
    assert(input.length == inputLength);
    var outs = new List<num>(outputLength);
    // for each outputneuron
    for(num row = 0; row < outputLength; row++){

      // Propagierungsfunktion: aufsummieren von allen (inputs * weights)
      var net = networth(input, row);

      // Aktivierungsfunktion
      var a = act(net);

      // Ausgabefunktion
      var o = a;
      outs[row] = o;
    }
    return outs;
  }
}