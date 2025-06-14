use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

pub struct CommandEntry {
    pub queue: Arc<vulkano::device::Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl CommandEntry {
    fn new(
        queue: Arc<vulkano::device::Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> CrystalResult<Self> {
        Ok(Self {
            queue,
            command_buffer_allocator,
        })
    }

    pub fn record_command_buffer<P>(
        &self,
        predicate: P,
    ) -> CrystalResult<Arc<PrimaryAutoCommandBuffer>>
    where
        P: Fn(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>),
    {
        let mut command_buffer_builder = match AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ) {
            Ok(builder) => builder,
            Err(e) => {
                log!("cannot create auto command buffer builder: {:?}", e);
                return Err(CrystalError::CannotCreateCommandManager);
            }
        };

        predicate(&mut command_buffer_builder);

        match command_buffer_builder.build() {
            Ok(command_buffer) => Ok(command_buffer.clone()),
            Err(e) => {
                log!("cannot build command buffer: {:?}", e);
                Err(CrystalError::CommandManagerError)
            }
        }
    }
}

pub struct CommandManager {
    pub graphics: Option<CommandEntry>,
}

impl CommandManager {
    pub fn new(
        queues: &[(Arc<vulkano::device::Queue>, vulkano::device::QueueFlags)],
    ) -> CrystalResult<Self> {
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            queues[0].0.device().clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let graphics = match queues
            .iter()
            .find(|(_, flags)| flags.intersects(vulkano::device::QueueFlags::GRAPHICS))
        {
            Some((queue, _)) => Some(CommandEntry::new(
                queue.clone(),
                command_buffer_allocator.clone(),
            )?),
            None => None,
        };

        Ok(Self { graphics })
    }
}
