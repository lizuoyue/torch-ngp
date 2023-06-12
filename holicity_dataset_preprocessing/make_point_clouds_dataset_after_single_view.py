import json
import os

if __name__ == "__main__":

    name_to_phase = {}
    for phase in ["train", "val", "test"]:
        with open(f"../data/holicity_point_cloud/point_clouds/{phase}.txt") as f:
            lines = [line.strip().split()[-1][:-4] for line in f.readlines()[1:]]
            for line in lines:
                name_to_phase[line] = phase

    with open("/cluster/project/cvg/zuoyue/torch-ngp/data/holicity_single_view/transforms.json") as f:
        transforms = json.load(f)
        for frame in transforms["frames"]:
            name = os.path.basename(frame["file_path"])[:22]
            if name in name_to_phase:
                phase = name_to_phase[name]
            else:
                continue
            frame["phase"] = phase
            frame["point_cloud"] = f"{phase}/{name}.npz"
    
    with open("../data/holicity_point_cloud/transforms.json", "w") as f:
        f.write(json.dumps(transforms, indent=2))
    