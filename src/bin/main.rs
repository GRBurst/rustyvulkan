extern crate vulkano;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};

fn main() {
    println!("Running main!");

    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance = Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");

    let physical = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

}
