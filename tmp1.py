from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn import MSELoss
from torch.optim import SGD
import hydra
import os
from tools_for_quant_offload.forward_hook import OffloadHookContext
from tools_for_quant_offload.graph_hook import OffloadSavedTensorHook
from class_dataset.myDataset import MyDataset
from hook.gradient_hook import get_record_gradient_hook
from class_model.longLinear import LongLinearModel
from my_utils.resouces_monitor import show_gpu_and_cpu_memory

os.environ["HYDRA_FULL_ERROR"] = "1"
record_dict = dict()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    print(cfg)
    model = LongLinearModel(n_layers=400).cuda()
    model.bfloat16()
    for n, p in model.named_parameters():
        p.register_hook(get_record_gradient_hook(model, record_dict=record_dict))
    dataset = MyDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=3)
    optimizer = SGD(model.parameters(), lr=1e-2)

    loss_fn = MSELoss()
    # loss_fn.register_forward_pre_hook(
    #     OffloadHookContext.get_align_device_pre_forward_hook(device="cuda", with_kwargs=True),
    #     with_kwargs=True,
    # )
    with OffloadHookContext(
            model=model,
            device="cuda",
            no_split_module_classes=["LlamaDecoderLayer", "GPT2TransformerBlock"],
            num_block=2,
            enable=True,
            with_backward_hook=False,
    ):
        with torch.autograd.graph.saved_tensors_hooks(
                pack_hook=OffloadSavedTensorHook.pack,
                unpack_hook=OffloadSavedTensorHook.unpack,
        ):
            for i in range(20):
                for x, y in dataloader:
                    print("before forward=======================================================")
                    show_gpu_and_cpu_memory()
                    x = x.to("cuda")
                    y = y.to("cuda")
                    x: torch.Tensor
                    y: torch.Tensor
                    # x = x.bfloat16()
                    # y = y.bfloat16()
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        o = model(x)
                        print(o.device)
                        print("before backward======================================================")
                        show_gpu_and_cpu_memory()
                        loss = loss_fn(o, y)
                        loss.backward()
                    loss: torch.Tensor
                    print(loss)


if __name__ == '__main__':
    main()
