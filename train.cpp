/*
    Copyright (c) 2013, TinyDNN example adopted to ANCA-cells (https://en.wikipedia.org/wiki/Anti-neutrophil_cytoplasmic_antibody)
	Class 0 is for P-ANCA, 1 is for C-ANCA

*/
#include <iostream>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/util.h"

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          tiny_dnn::core::backend_t backend_type) {
  // construct nets
  //
  // C : convolution
  // S : sub-sampling
  // F : fully connected
  // clang-format off
  using fc = tiny_dnn::layers::fc;
  using conv = tiny_dnn::layers::conv;
  using ave_pool = tiny_dnn::layers::ave_pool;
  using tanh = tiny_dnn::activation::tanh;
  using relu = tiny_dnn::activation::relu;

  using tiny_dnn::core::connection_table;
  using padding = tiny_dnn::padding;


  nn << conv(32, 32, 5, 1, 2) << relu() // 32x32in, conv3x3
	  << ave_pool(28, 28, 2, 2) << relu()  // 28x28in, pool2x2
	  << fc(14 * 14 * 2, 120) << relu()
	  << fc(120, 2);

}

static void train_lenet(const std::string &data_dir_path,
                        double learning_rate,
                        const int n_train_epochs,
                        const int n_minibatch,
                        tiny_dnn::core::backend_t backend_type) {
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  tiny_dnn::adagrad optimizer;

  construct_net(nn, backend_type);

  std::cout << "load models..." << std::endl;
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;
  

	std::fstream ifs;
	ifs.open("C:/tmp/train.csv", std::ifstream::in);

	while (ifs.good())
	{
	int l;
	ifs >> l;
	train_labels.push_back(l);
	tiny_dnn::vec_t d;
	for (int i = 0; i < 32 * 32; i++)
	{
		float v;
		ifs >> v;
		d.push_back(v / 127. - 1.0);
	}
	train_images.push_back(d);
	}

	std::fstream ifs1;
	ifs1.open("C:/tmp/test.csv", std::ifstream::in);

	while (ifs1.good())
	{
		int l;
		ifs1 >> l;
		test_labels.push_back(l);
		tiny_dnn::vec_t d;
		for (int i = 0; i < 32 * 32; i++)
		{
			float v;
			ifs1 >> v;
			d.push_back( v / 127. -1.0 );
		}
		test_images.push_back(d);
	}

	test_labels = train_labels;
	test_images = train_images;


  std::cout << "start training" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  optimizer.alpha *=
    std::min(tiny_dnn::float_t(4),
             static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
                          n_train_epochs, on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);
  tiny_dnn::vec_t result = nn.predict(test_images[0]);
  for each (auto var in result)
  {
	  std::cout << var << std::endl;
  }
  // save network model & trained weights
  nn.save("LeNet-model");

}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {{
    "internal", "nnpack", "libdnn", "avx", "opencl",
  }};
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<tiny_dnn::core::backend_t>(i);
    }
  }
  return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
            << " --learning_rate 1"
            << " --epochs 30"
            << " --minibatch_size 16"
            << " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
  double learning_rate                   = 1;
  int epochs                             = 20;
  std::string data_path                  = "";
  int minibatch_size                     = 16;
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }

  std::cout << "Running with the following parameters:" << std::endl
            << "Data path: " << data_path << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
	  epochs = 20;
	  minibatch_size  = 10;
    train_lenet(data_path, learning_rate, epochs, minibatch_size, backend_type);
	
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
  system("pause");
  return 0;
}
