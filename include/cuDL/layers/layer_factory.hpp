#ifndef LAYER_FACTORY_H_
#define LAYER_FACTORY_H_

#include <boost/shared_ptr.hpp>
#include <map>
#include "cuDL/layers/layer_paras.hpp"
#include "cuDL/core/dcon.hpp"
#include "cuDL/layers/layer.hpp"
#include "cuDL/core/base.hpp"
#include <fstream>

using namespace std;

template <typename T>
class Layer;

template <typename T>
class LayerRegistry {
	public:
		typedef boost::shared_ptr<Layer<T> > (*Creator)(const LayerParas&);
		typedef map<string, Creator> CreatorRegistry;

		static CreatorRegistry& Registry() {
			static CreatorRegistry* g_registry_ = new CreatorRegistry();
			return *g_registry_;
		}

		static void AddCreator(const string& type, Creator creator) {
		    CreatorRegistry& registry = Registry();
		    registry[type] = creator;
		}

		static boost::shared_ptr<Layer<T> > CreateLayer(const LayerParas& paras) {
		    const string& type = paras.type;
		    CreatorRegistry& registry = Registry();
		    return registry[type](paras);
		}

	private:
		LayerRegistry() {}
};


template <typename T>
class LayerRegisterer {
	public:
		LayerRegisterer(const string& type,
		                boost::shared_ptr<Layer<T> > (*creator)(const LayerParas&)) {
			LayerRegistry<T>::AddCreator(type, creator);
		}
};

#define REGISTER_LAYER_CREATOR(type, creator)                                    \
	static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
	static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                               \
	template <typename T>                                                        \
	boost::shared_ptr<Layer<T> > Creator_##type##Layer(const LayerParas& paras)  \
	{                                                                            \
		return boost::shared_ptr<Layer<T> >(new type##Layer<T>(paras));          \
	}                                                                            \
	REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

#endif //LAYER_FACTORY_H_