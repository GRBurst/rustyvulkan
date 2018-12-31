#[macro_use]
extern crate vulkano;

use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;


fn main(){
println!("Hello, world!");

let instance = Instance::new(None, &InstanceExtensions::none(), None)
    .expect("failed to create instance");

}
