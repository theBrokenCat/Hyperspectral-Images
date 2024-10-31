import numpy as np
from typing import Any
from .blocks import *

class PreprocessingPipeline:
    def __init__(self, data: np.ndarray) -> None:
        self.original_data = data.copy()

    def _analyze_data(self, data: np.ndarray) -> None:
        # compute the mean and std of the data
        self.mean = np.mean(data)
        self.std = np.std(data)
        # print the mean and std of the data
        print(f"Mean: {self.mean}")
        print(f"Std: {self.std}")

    def __call__(
        self, pipeline: list[GenericPipelineBlock], verbose: bool = True, plot_masks: bool = False, mask=None
    ) -> Any:
        """
        Executes the preprocessing pipeline on the original data.
        Args:
            pipeline (list[GenericPipelineBlock]): The list of pipeline blocks to be executed.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            plot_masks (bool, optional): Whether to plot the masks. Defaults to False.
            mask (np.ndarray, optional): The mask to be used in the pipeline. Defaults to None. If provided, the mask will be used and the pipeline will not create a new mask.
        Returns:
            Any: The processed data and the full mask. The full mask is the logical AND of all the masks, pixels that are False in the full mask are not considered in the unmixing process.
        Raises:
            TypeError: If the pipeline is not a list of GenericPipelineBlock.
        """
        previous_mask = mask

        data = np.copy(self.original_data)
        if verbose:
            print("--------------Original data--------------")
            print(f" - Data shape: {data.shape}")
            print(f" - Data type: {data.dtype}")
            print(f" - Data range: {np.min(data)} - {np.max(data)}")
            print(f" - Data mean: {np.mean(data)} - Data std: {np.std(data)}")
            print("--------------Running Pipeline--------------")
        if not isinstance(pipeline, list):
            raise TypeError("Pipeline must be a list of GenericPipelineBlock")

        # First we are going to find the blocks that return a mask
        masking_blocks = [b for b in pipeline if b.type == "Mask"]
        # Then we are going to find the blocks that return a processed version of the data
        #processing_blocks = [b for b in pipeline if b.type == "Processor" and str(b) != "Denoise"]
        processing_blocks = [b for b in pipeline if b.type == "Processor"]

        # LEGACY: Denoise takes priority over the other blocks, so we are going to run it first
        # This way, when masking, the algorithms are expected to be more accurate
        # if "Denoise" in [str(b) for b in pipeline]:
        #     print(" - Running pipeline block: Denoise")
        #     data, noise_corr = ProcessDenoise()(data)

        masks = []
        full_mask = np.ones((data.shape[0], data.shape[1]), dtype=bool)

        # We are going to run the masking blocks first
        if previous_mask is not None:
            full_mask = previous_mask
            print(" - Using previous mask, skipping masking blocks")
        else:
            for block in masking_blocks:
                if verbose:
                    print(" - Running pipeline block: " + str(block))
                if not isinstance(block, GenericPipelineBlock):
                    raise TypeError("Pipeline element" + str(block) + " must be child of GenericPipelineBlock")

                mask = block(data)
                masks.append(mask)
                full_mask = full_mask & mask

        # Now we are going to run the processing blocks
        for block in processing_blocks:
            if verbose:
                print(" - Running pipeline block: " + str(block))
            if not isinstance(block, GenericPipelineBlock):
                raise TypeError("Pipeline element" + str(block) + " must be child of GenericPipelineBlock")
            
            print(f"   + Input Data shape: {data.shape}")
            data = block._apply_operation_to_masked(data, full_mask, block)
            print(f"   + Output Data shape: {data.shape}")
        
        # Print final information on the data
        if verbose:
            masked_pixels = full_mask.size - np.sum(full_mask)
            total_pixels = full_mask.size
            print("--------------Final data--------------")
            print(f" - Data shape: {data.shape}")
            print(f" - Data type: {data.dtype}")
            print(f" - Data range: {np.min(data)} - {np.max(data)}")
            print(f" - Data mean: {np.mean(data)} - Data std: {np.std(data)}")
            print(f" - Masked pixels {masked_pixels} of {total_pixels} ({masked_pixels/total_pixels*100:.2f}%)")
            print("--------------Pipeline finished--------------")

        if plot_masks:
            if len(masking_blocks) == 0:
                print("No masks to plot")
                return data, full_mask, masks
            # TODO change the visualization to be 1 image in which each color is a different mask, with labels
            fig, axs = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
            try:
                a = axs[0].imshow(full_mask, vmin=0, vmax=1, cmap=plt.cm.gray)
                axs[0].set_title("Full mask", fontsize=9)
            except:
                axs.imshow(full_mask, vmin=0, vmax=1, cmap=plt.cm.gray)
                axs.set_title("Full mask", fontsize=9)
            for i, mask in enumerate(masks):
                axs[i + 1].imshow(mask, vmin=0, vmax=1, cmap=plt.cm.gray)
                axs[i + 1].set_title(str(masking_blocks[i]).removeprefix("Mask"), fontsize=8)
                axs[i + 1].axis("off")
            # plt.colorbar(a)
            # plt.suptitle("Masks from the preprocessing pipeline")
            plt.show()
        return data, full_mask, masks
