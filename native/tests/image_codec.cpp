#include "image.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
int main(){try{
 d5server::Image image;image.width=32;image.height=32;image.channels=4;
 image.rgb.resize(32*32*4);
 for(size_t i=0;i<32*32;++i){image.rgb[4*i]=float((i*7+51)%256)/255;image.rgb[4*i+1]=float((i*13+97)%256)/255;image.rgb[4*i+2]=float((i*31+211)%256)/255;image.rgb[4*i+3]=float(i%256)/255;}
 auto decoded=d5server::decode_image(d5server::encode_png(image));
 if(decoded.width!=32||decoded.height!=32||decoded.rgb.size()!=image.rgb.size())throw std::runtime_error("Image shape changed");
 float worst=0;for(size_t i=0;i<image.rgb.size();++i)worst=std::max(worst,std::abs(image.rgb[i]-decoded.rgb[i]));
 if(worst>1e-6f)throw std::runtime_error("RGBA PNG round-trip changed pixel channels");
 std::cout<<"RGBA_png_round_trip_OK max="<<worst<<" (asymmetric RGB and varying straight alpha)\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
