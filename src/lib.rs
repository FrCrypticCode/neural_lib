#![no_main]
//! # Neural Lib
//! In Progress !!! -> Some fixes in load  
//! This is a small library to make group of neurons.
//! Every group is create with own parameters and the functions required to make predictions and training

/// A Neuron is defined by a weight and an impact. Every value passed in is change by the property of the neuron
struct Neuron{
    weight: f64,
    bias:f64
}
impl Default for Neuron{
    fn default() -> Self {
        return Neuron{weight:0.5,bias:0.5}
    }
}
impl Neuron{
    fn new(weight:f64,bias:f64)->Self{
        return Neuron{weight:weight,bias:bias}
    }
}

/// Main Struct to make a Layer of neural network
/// It contains few Neuron struct and the activation function for them
pub struct Layer{
    neurons:Vec<Neuron>,
    act:Activate
}
impl Layer{
    /// Add layer, need to a vec of Neurons parameters and type of function
    pub fn new(fnact:&Activate,param:&Vec<(f64,f64)>)->Self{ 
        let mut network = Layer{
            neurons: Vec::with_capacity(param.len() as usize),
            act:fnact.clone()
        };
        for i in param{
            network.neurons.push(Neuron::new(i.0,i.1));
        }
        return network
    }
    /// Add layer, need to define number of neurons and type of function
    pub fn default(x:u32,fnact:&Activate) -> Self {
        let mut layer = Layer{
            neurons: Vec::with_capacity(x as usize),
            act:fnact.clone()
        };
        for _i in 0..x{
            layer.neurons.push(Neuron::new(0.5,0.5));
        }
        return layer
    }
    /// Calculate the Weighted Sum 
    fn sum_pond(&self,x:&Vec<i32>)->f64{
        let mut sum:f64 = 0.0;
        for (id,n) in self.neurons.iter().enumerate(){
            sum += x[id] as f64 * n.weight + n.bias; 
        }
        return sum
    }
}

#[derive(Clone)]
/// Enum to choose your Activation function, by default => Sigmoid function
pub enum Activate{  // Choose your function to activate neurons
    Sig,
    Rel,
    Tan
}
impl Default for Activate{  // Default Status => Sigmoid function
    fn default()->Self{
        return Activate::Sig
    }
}

fn activate(net:&Layer,x:&Vec<i32>, act:Activate)->Result<f64,String>{
    let sum = net.sum_pond(x);
    match act{
        Activate::Sig=>{return Ok(sigmoid(sum))},
        Activate::Rel=>{return Ok(relu(sum))},
        Activate::Tan=>{return Ok(tanh(sum))}
    }

}

fn sigmoid(x:f64)->f64{
    return 1.0/(1.0+(-x).exp())
} 

fn relu(x:f64)->f64{
    if x > 0.0{
        return x
    }
    else{
        return 0.0
    }
}

fn tanh(x:f64)->f64{
    return ((2.0 * x).exp()-1.0) / ((2.0 * x).exp()+1.0)
}

/// Define a Neural Network
struct Network{
    layers:Vec<Layer>
}
impl Network{
    fn new(nb_lay:usize,lays_params:Vec<Vec<(f64,f64)>>,lays_fn:Vec<Activate>)->Network{
        let mut lays:Vec<Layer> = Vec::new();
        for l in 0..nb_lay{
            lays.push(Layer::new(&lays_fn[l], &lays_params[l]));
        }
        return Network { layers: lays};
    }
    fn default(nb_lay:usize,nb_n:u32,lays_fn:Vec<Activate>)->Network{
        let mut lays:Vec<Layer> = Vec::new();
        for l in 0..nb_lay{
            lays.push(Layer::default(nb_n,&lays_fn[l]));
        }
        return Network { layers: lays};
    }
    /*
    /// Initiate a training session
    /// You can specifie the inputs and the number of repetitions from the nb variable
    pub fn train(&mut self, inputs:&Vec<Vec<i32>>,targets:&Vec<f64>,learning_rate:f64,nb:usize){
        for _ in 0..nb{
            for (input,target) in inputs.iter().zip(targets.iter()){
                self.train_single(input, *target, learning_rate);
            }
        }
    }

    fn train_single(&mut self, input:&Vec<i32>, target:f64, learning_rate:f64){
        let output = self.prediction(input,self.act.clone()).unwrap();
        let error = target - output;
        for (neuron,x) in self.neurons.iter_mut().zip(input.iter()){
            neuron.weight += learning_rate * error * output * (1.0-output) * *x as f64;
            neuron.bias += learning_rate * error * output * (1.0-output);
        }
    }

    /// Launching a prediction with values -> This function can be call manually or by the training session
    pub fn prediction(&self, input:&Vec<i32>, act:Activate)->Result<f64,String>{   // Call a result on an entry
        match activate(self,input,act){
            Ok(x)=>{return Ok(x)},
            Err(err)=>{return Err(err)}
        }
    } */
}

enum Class{
    Binary,
    MultiClass
}