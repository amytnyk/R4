#ifndef R4_WORLD_HPP
#define R4_WORLD_HPP

#include <string>
#include <sstream>
#include <vector>
#include <charconv>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <view.hpp>
#include <light.hpp>
#include <objects/sphere.hpp>
#include <objects/triangle.hpp>
#include <objects/parallelpiped.hpp>
#include <objects/tetrahedron.hpp>
#include <objects/entity_list.hpp>

void attribute_parse(Material &material, std::stringstream &str_stream) {
    std::string key;
    str_stream >> key;
    while (key != ")") {
        if (key == "reflect")
            str_stream >> material.reflect;
        else if (key == "shine")
            str_stream >> material.shine;
        else if (key == "transpar")
            str_stream >> material.transparent;
        else if (key == "specular")
            str_stream >> material.specular;
        else if (key == "ambient")
            str_stream >> material.ambient;
        else if (key == "diffuse")
            str_stream >> material.diffuse;
        str_stream >> key;
    }
}

template<typename VecType>
struct World {
    Color ambient{};
    Color background{};
    View<VecType> view;
    Vector<Light<VecType>> lights;
    EntityList<VecType> entities;

    VecType::value_type max_depth;

    __host__ explicit World(std::string data) {
        for (auto &c: data)
            c = static_cast<char>(std::tolower(c));

        std::replace(data.begin(), data.end(), '[', ' ');
        std::replace(data.begin(), data.end(), ']', ' ');
        std::replace(data.begin(), data.end(), '{', ' ');
        std::replace(data.begin(), data.end(), '}', ' ');
        data = std::regex_replace(data, std::regex("{"), "( ");
        data = std::regex_replace(data, std::regex(")"), " )");

        std::stringstream str_stream(data);
        std::string key;
        typename VecType::value_type value;
        std::string str_value;

        std::unordered_map<std::string, Material> attr_dict;
        std::string last_attr;
        typename VecType::value_type last_radius;
        Color last_color;

        World<VecType> world;

        while (str_stream >> key) {
            if (key == ">") {
                std::string empty;
                std::getline(str_stream, empty);
            } else if (key == "ambient")
                str_stream >> world.ambient;
            else if (key == "background")
                str_stream >> world.background;
            else if (key == "maxdepth")
                str_stream >> world.max_depth;
            else if (key == "view") {
                while (key != ")") {
                    if (key == "from") {
                        str_stream >> world.view.direction.origin();
                    } else if (key == "to") {
                        str_stream >> world.view.direction.direction();
                        world.view.direction.direction() -= world.view.direction.origin();
                    } else if (key == "up")
                        str_stream >> world.view.up;
                    else if (key == "over")
                        str_stream >> world.view.over;
                    else if (key == "angle")
                        str_stream >> world.view.angle;
                    str_stream >> key;
                }
            } else if (key == "light") {
                Light<VecType> cur_light;
                while (key != ")") {
                    if (key == "direction") {
                        cur_light.type = Light<VecType>::DIRECTIONAL;
                        str_stream >> cur_light.direction;
                    } else if (key == "position") {
                        cur_light.type = Light<VecType>::POSITIONAL;
                        str_stream >> cur_light.position;
                    } else if (key == "color")
                        str_stream >> last_color;
                    str_stream >> key;
                }
                cur_light.color = last_color;
                world.lights.push(cur_light);
            } else if (key == "attributes") {
                Material material;
                str_stream >> last_attr;
                attribute_parse(material, str_stream);
                attr_dict[last_attr] = material;
            } else if (key == "sphere") {
                VecType center;
                Material material;
                bool found_attr = false;
                while (key != ")") {
                    if (key == "center")
                        str_stream >> center;
                    else if (key == "radius")
                        str_stream >> last_radius;
                    else if (key == "attributes") {
                        found_attr = true;
                        str_stream >> key;
                        if (key == "(")
                            attribute_parse(material, str_stream);
                        else
                            material = attr_dict[key];
                    }
                    str_stream >> key;
                }
                if (!found_attr)
                    material = attr_dict[last_attr];
                world.entities.addChild(new Sphere{center, last_radius, material});
            } else if (key == "parallelpiped") {
                Array<VecType, VecType::dim> dots;
                Material attr;
                bool found_attr = false;
                while (key != ")") {
                    if (key == "vertices") {
                        for (int i = 0; i < VecType::dim; ++i)
                            str_stream >> dots[i];
                    } else if (key == "attributes") {
                        found_attr = true;
                        str_stream >> key;
                        if (key == "(")
                            attribute_parse(attr, str_stream);
                        else
                            attr = attr_dict[key];
                    }
                    str_stream >> key;
                }
                if (!found_attr)
                    attr = attr_dict[last_attr];
                world.entities.addChild(new Parallelpiped<VecType>(dots, attr));
            } else if (key == "tetrahedron") {
                Array<VecType, VecType::dim> dots;
                Material material;
                bool found_attr = false;
                while (key != ")") {
                    if (key == "vertices") {
                        for (int i = 0; i < VecType::dim; ++i)
                            str_stream >> dots[i];
                    } else if (key == "attributes") {
                        found_attr = true;
                        str_stream >> key;
                        if (key == "(")
                            attribute_parse(material, str_stream);
                        else
                            material = attr_dict[key];
                    }
                    str_stream >> key;
                }
                if (!found_attr)
                    material = attr_dict[last_attr];
                world.entities.addChild(new Tetrahedron<VecType>(dots, material));
            }
        }
    }
};

#endif //R4_WORLD_HPP
