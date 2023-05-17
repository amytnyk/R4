#ifndef R4_SCENE_HPP
#define R4_SCENE_HPP

#include <string>
#include <fstream>
#include <camera.hpp>
#include <world.hpp>
#include <objects/entity.hpp>
#include <sstream>

template<typename VecType>
class Scene {
public:
    World<VecType> world;

    explicit Scene(const std::string &path) : world(read_file(path)) {}

    __host__ __device__ Color rayTrace(const Vec3d &relative_direction,
                                       const Camera<VecType> &camera) {
        return rayTrace(camera.ray_to(relative_direction));
    }
private:

    std::string read_file(const std::string &path) {
        std::ifstream in{path};
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    __host__ __device__ Color rayTrace(Ray<VecType> ray, size_t level = 0) {
        ray = ray.forward();

        Hit<VecType> hit_record;
        if (!world.entities.hit(ray, hit_record))
            return world.background;

        Color color = hit_record.material->ambient * world.ambient;
        typename VecType::value_type n_dot_d;

        if (hit_record.material->diffuse.present() || hit_record.material->specular.present()) {
            n_dot_d = hit_record.normal.dot(ray.direction());

            if (n_dot_d > 0) {
                hit_record.normal *= -1;
                n_dot_d *= -1;
            }

            auto intersection = Ray<VecType>{hit_record.point, hit_record.normal}.forward().origin();

            for (size_t i = 0; i < world.lights.size(); ++ i) {
                const auto &light = world.lights[i];
                VecType light_direction;
                typename VecType::value_type min_dist;

                if (light.type == Light<VecType>::DIRECTIONAL) {
                    light_direction = light.direction;
                } else {
                    light_direction = light.position - hit_record.point;
                    typename VecType::value_type norm = min_dist = light_direction.norm();
                    if (norm < FLT_EPSILON)
                        light_direction = hit_record.normal;
                    else
                        light_direction /= norm;
                }

                Color light_color = light.color;

                world.entities.call_for_hits(Ray<VecType>(hit_record.point, light_direction),
                                             [&light_color](const Entity<VecType> &entity) {
                                                 light_color *= entity.material().transparent;
                                             });

                if (!light_color.present())
                    continue;

                auto tmp = hit_record.normal.dot(light_direction);
                if (tmp <= 0)
                    continue;

                color += tmp * hit_record.material->diffuse * light_color;

                if (hit_record.material->specular.present()) {
                    auto refl = 2 * tmp * hit_record.normal - light_direction;
                    tmp = -refl.dot(ray.direction());
                    if (tmp > 0)
                        color +=
                                std::pow(tmp, hit_record.material->shine) * hit_record.material->specular * light_color;
                }
            }
        }

        if (world.max_depth && (level == world.max_depth))
            return color;

        if (hit_record.material->transparent.present()) {
            auto direction = n_dot_d * hit_record.normal + ray.direction();
            color += rayTrace(Ray<VecType>(hit_record.point, direction), ++level)
                     * hit_record.material->transparent;
        }

        if (hit_record.material->reflect) {
            auto reflection_direction = ray.direction() - 2 * n_dot_d * hit_record.normal;
            color += rayTrace(Ray<VecType>(hit_record.point, reflection_direction), ++level)
                     * hit_record.material->specular;
        }
        return color;
    }
};

#endif //R4_SCENE_HPP
