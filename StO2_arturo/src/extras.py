from hypyflow import blocks
#from src.utils import ProcessElasticNet

class ProcessChains:
    pc1 = [
            blocks.MaskNegativeValues(),
            blocks.MaskZeroValues(),
            blocks.MaskSaturatedValues(99.5), #
            #blocks.ProcessNormalize(type="MinMax"),#
            #blocks.ProcessDenoise(),
            blocks.MaskRxAnomalyDetection(0.2), #
            #blocks.MaskTargetSignature(interactive=True,select_area=True,method="SID"), #TODO: hacer que la barra selectora sea logarítmica, para que cuando me acerque a 0.1 no sea tan dificil seleccionar
            blocks.MaskTargetSignature(interactive=True,select_area=True,method="SAM"),
            blocks.ProcessSmoothSpectral("gaussian",  {"sigma": 1}),#(method="moving_average", params={"window_size": 3}),
            #blocks.ProcessSmoothSpatial(),
            blocks.MaskZeroValues(),
            blocks.MaskNegativeValues()
    ]    
    pc11 = [
            blocks.MaskNegativeValues(),
            blocks.MaskZeroValues(),
            blocks.MaskSaturatedValues(99.99), #
            #blocks.ProcessNormalize(type="MinMax"),#
            #blocks.ProcessDenoise(),
            blocks.MaskRxAnomalyDetection(0.2), #
            #blocks.MaskTargetSignature(interactive=True,select_area=True,method="SID"), #TODO: hacer que la barra selectora sea logarítmica, para que cuando me acerque a 0.1 no sea tan dificil seleccionar
            blocks.MaskTargetSignature(interactive=True,select_area=True,method="SAM"),
            blocks.ProcessSmoothSpectral("gaussian",  {"sigma": 1}),#(method="moving_average", params={"window_size": 3}),
            #blocks.ProcessSmoothSpatial(),
            blocks.MaskZeroValues(),
            blocks.MaskNegativeValues()
        ]
    pc11 = [
            blocks.MaskNegativeValues(),
            blocks.MaskZeroValues(),
            blocks.MaskSaturatedValues(99.99),
            blocks.MaskRxAnomalyDetection(0.2),
            blocks.ProcessNormalize("MinMax"),
            blocks.MaskTargetSignature(interactive=True, select_area=True, method="SAM"),
            blocks.ProcessSmoothSpectral(method = "gaussian", params = {"sigma": 1})
        ]
    pc2 = [
            blocks.MaskNegativeValues(),
            blocks.MaskZeroValues(),
            blocks.MaskSaturatedValues(99.99),
            blocks.MaskRxAnomalyDetection(0.2),
            blocks.ProcessNormalize("MinMax"),
            blocks.MaskTargetSignature(interactive=True, select_area=True, method="SAM"),
            blocks.ProcessSmoothSpectral(method = "savitzky_golay",params = {"window_size": 3, "order": 1})
        ]
    pc3 = [
            blocks.MaskNegativeValues(),
            blocks.MaskZeroValues(),
            blocks.MaskSaturatedValues(99.99),
            blocks.MaskRxAnomalyDetection(0.2),
            blocks.ProcessNormalize("MinMax"),
            blocks.MaskTargetSignature(interactive=True, select_area=True, method="SAM"),
            blocks.ProcessSmoothSpectral(method = "savitzky_golay",params = {"window_size": 3, "order": 1}),
            blocks.ProcessSmoothSpectral(method = "savitzky_golay",params = {"window_size": 3, "order": 1})
        ]
    pc4 = [
            blocks.MaskNegativeValues(),
            blocks.MaskZeroValues(),
            blocks.MaskSaturatedValues(99.99),
            blocks.MaskRxAnomalyDetection(0.2),
            blocks.ProcessNormalize("MinMax"),
            blocks.MaskTargetSignature(interactive=True, select_area=True, method="SAM"),
            blocks.ProcessSmoothSpectral(method = "savitzky_golay",params = {"window_size": 3, "order": 1}),
            blocks.ProcessDenoise()
        ]
    pc5 = [
            blocks.MaskNegativeValues(),
            blocks.MaskZeroValues(),
            blocks.MaskSaturatedValues(99.99),
            blocks.MaskRxAnomalyDetection(0.2),
            blocks.ProcessNormalize("MinMax"),
            blocks.MaskTargetSignature(interactive=True, select_area=True, method="SAM"),
            blocks.ProcessSmoothSpectral(method = "savitzky_golay",params = {"window_size": 3, "order": 1}),
            #blocks.ProcessReduceDimensionality("PCA"),  # Assuming "reduction_method" is the correct parameter
            blocks.ProcessDenoise()
        ]
    pc6 = [
            blocks.MaskNegativeValues(),
            blocks.MaskZeroValues(),
            blocks.MaskSaturatedValues(99.99),
            blocks.MaskRxAnomalyDetection(0.2),
            blocks.ProcessNormalize("MinMax"),
            blocks.MaskTargetSignature(interactive=True, select_area=True, method="SAM"),
            blocks.ProcessSmoothSpectral(method = "savitzky_golay",params = {"window_size": 3, "order": 1}),
            #blocks.ProcessReduceDimensionality("PCA"),
            blocks.ProcessDenoise()
            #ProcessElasticNet(alpha=0.1, l1_ratio=0.7)
        ]
    
    pc7 = [
    blocks.MaskNegativeValues(),
    blocks.MaskZeroValues(),
    blocks.MaskSaturatedValues(99.99), #
    #blocks.ProcessNormalize(type="MinMax"),#
    #blocks.ProcessDenoise(),
    blocks.MaskRxAnomalyDetection(0.2), #
    #blocks.MaskTargetSignature(interactive=True,select_area=True,method="SID"), #TODO: hacer que la barra selectora sea logarítmica, para que cuando me acerque a 0.1 no sea tan dificil seleccionar
    blocks.MaskTargetSignature(interactive=True,select_area=True,method="SAM"),
    blocks.ProcessSmoothSpectral("gaussian",  {"sigma": 1}),#(method="moving_average", params={"window_size": 3}),
    #blocks.ProcessSmoothSpatial(),
    blocks.MaskZeroValues(),
    blocks.MaskNegativeValues()
    ]