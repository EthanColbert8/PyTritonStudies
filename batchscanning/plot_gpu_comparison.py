import plotting

save_folder = "/depot/cms/users/colberte/SONIC/Scans/comparisons"
save_name = "T4_V100_A30_A100_MI100_particlenet_comparison_new.pdf"

######## ParticleNet (ONNX) ########
T4_info = {
    'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'throughput': [1774.6310026947292, 2671.571525158178, 3585.9798220905773, 4208.995195913258, 4486.510587466162, 4624.3225366549095, 4698.801781719015, 4565.7845139798, 4462.443130062966, 4365.546925197751],
    'latency': [2.7126661899999998, 3.52814765, 4.926460680000002, 7.87915253, 14.419367000000005, 27.724642090000003, 54.51585286, 112.17653047, 229.47688458000005, 469.14526071000006],
    'name': "Nvidia T4",
    'color': '#e42536',
}
V100_info = {
    'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'throughput': [1510.4324673604674, 1976.772034535826, 4511.602330175681, 7690.026726645595, 10057.256900815453, 11589.76495861704, 12490.551072096783, 13044.83240090102, 13305.15483959488, 13481.231785972468],
    'latency': [3.26680558, 4.39730897, 4.55558036, 4.712980520000001, 6.67730302, 11.314755810000001, 20.690159360000003, 39.296632030000005, 77.01334716, 151.91495257],
    'name': "Nvidia V100 (32GB)",
    'color': '#f89c20',
}
A100_info = {
    'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'throughput': [1058.7931533448316, 2224.0555881046803, 4088.732850126853, 10016.732861497665, 15153.182608816096, 19771.226978716448, 21023.0769969828, 20545.623661493337, 23905.871469594465, 24385.810948535935],
    'latency': [3.8139993400000005, 3.63479743, 5.02017831, 4.36249963, 4.954942910000001, 6.880795280000002, 12.39552062, 25.158163879999996, 42.887490299999996, 84.00423312000001],
    'name': "Nvidia A100 (40 GB)",
    'color': '#964a8b',
}
A30_info = {
    'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'throughput': [1821.493627465826, 3252.883241722367, 5609.682440345116, 7959.963820672067, 10383.550566583737, 11860.708718903714, 12867.067038378069, 13484.351523067384, 13760.83339910296, 13662.378458194153],
    'latency': [2.7772505499999998, 3.1646343999999997, 3.4877540899999997, 4.439695890000001, 6.3659496, 10.938322880000001, 20.0222377, 37.99050286, 74.42816869, 149.90151993],
    'name': "Nvidia A30",
    'color': '#9c9ca1',
}
MI100_info = {
    'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'throughput': [2681.3040815924473, 4463.817178540191, 6946.649079329508, 9764.879291008063, 12186.972639940745, 13802.076000608022, 15367.685339636451, 16312.400330599681, 16391.58472761094, 16456.10702138114],
    'latency': [1.49993855, 1.79237202, 2.30443229, 3.281689270000001, 5.260859460000001, 9.280878660000003, 16.663216100000003, 31.387745879999997, 62.47629806, 124.45384196],
    'name': "AMD MI100",
    'color': '#5790fc',
}

######## DeepMET (Tensorflow) ########
# T4_info = {
#     'batch_size': ,
#     'throughput': ,
#     'latency': ,
#     'name': "Nvidia T4",
#     'color': 'darkred',
# }
# V100_info = {
#     'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
#     'throughput': [2464.759178213887, 3825.281571377001, 4602.2848992089475, 5064.925755396175, 6060.187669199425, 5100.191703116571, 4017.23881070375, 4047.368978889214, 3781.512044269236, 3763.659179175782],
#     'latency': [1.6747565000000002, 2.09267325, 3.4838879000000005, 6.332740300000001, 10.5998643, 25.10056335, 63.729593050000005, 126.53249195000001, 271.17068830000005, 544.5249685000001],
#     'name': "Nvidia V100 (32GB)",
#     'color': 'orange',
# }
# A100_info = {
#     'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
#     'throughput': [2817.761531883449, 4511.2275548634925, 5768.809462672752, 6044.792148265804, 6219.372263063512, 6150.0572123085585, 6488.263047486394, 6797.66212285748, 6973.40579051587, 7150.074113912195],
#     'latency': [1.435203, 1.78215635, 2.79151885, 5.310565200000001, 10.324819799999998, 21.1117829, 39.460406600000006, 75.43292985000001, 146.84672515000003, 286.44170745],
#     'name': "Nvidia A100 (40 GB)",
#     'color': 'fuchsia',
# }
# A30_info = {
#     'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
#     'throughput': [2232.1693372378986, 3325.1407090551206, 3767.2245884111353, 4158.898379596854, 4258.52586190397, 3900.872057520291, 3579.1840484326735, 3738.731778804389, 3880.4272114148857, 3618.538277379395],
#     'latency': [1.8231725, 2.42476585, 4.3121638, 7.721611350000001, 15.0998931, 32.9642884, 71.53863285, 136.96388305, 264.6003148, 566.0707449],
#     'name': "Nvidia A30",
#     'color': 'green',
# }
# MI100_info = {
#     'batch_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
#     'throughput': [2455.5354578382917, 3864.9926861712756, 5413.723800625871, 6825.023087540059, 8244.963417705902, 7822.402807689051, 5470.499247909242, 5457.151713264117, 5634.086779318604, 5641.969681535328, 5087.9729982234785],
#     'latency': [1.66214687, 2.0876048000000003, 2.9761427200000004, 4.699904310000001, 7.797242460000002, 16.507140480000004, 47.47595453000001, 93.83795975999999, 181.82255172, 363.41830846000005, 805.05750935],
#     'name': "AMD MI100",
#     'color': 'blue',
# }

plot_dict = {
    T4_info['name']: T4_info,
    V100_info['name']: V100_info,
    A30_info['name']: A30_info,
    A100_info['name']: A100_info,
    MI100_info['name']: MI100_info,
}

plotting.plot_throughput_latency(plot_dict, save_folder+"/"+save_name, ratio_key=MI100_info['name'], ratio_label="All / MI100")