#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

const double PI = 3.14159265358979323846;

struct Vec2 {
    double x, y;
    Vec2(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    Vec2 operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
    Vec2 operator*(double s) const { return Vec2(x * s, y * s); }
    double norm() const { return std::sqrt(x*x + y*y); }
    double dot(const Vec2& v) const { return x*v.x + y*v.y; }
};

struct Stress {
    double sxx, syy, sxy;
    Stress(double sxx_ = 0, double syy_ = 0, double sxy_ = 0) 
        : sxx(sxx_), syy(syy_), sxy(sxy_) {}
    Stress operator+(const Stress& s) const {
        return Stress(sxx + s.sxx, syy + s.syy, sxy + s.sxy);
    }
};

struct Dislocation {
    Vec2 pos;
    Vec2 b;
    int crack_id;
    bool is_positive;
};

struct Crack {
    Vec2 center;
    double length;
    double angle;
    std::vector<int> dipole_indices;
    double resistance;
};

class SimpleCrackSystem {
private:
    std::vector<Dislocation> dislocations;
    std::vector<Crack> cracks;
    Stress external_stress;
    double nu = 0.3;
    double G = 80e9;
    double b = 2.5e-10;
    
public:
    void setExternalStress(double sxx, double syy, double sxy) {
        external_stress = Stress(sxx, syy, sxy);
    }
    
    void initializeCracks(int num, double min_len, double max_len, 
                         double domain, unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> pos_dist(-domain/2, domain/2);
        std::uniform_real_distribution<> len_dist(min_len, max_len);
        std::uniform_real_distribution<> ang_dist(0, 2*PI);
        
        for (int i = 0; i < num; ++i) {
            Crack crack;
            crack.center = Vec2(pos_dist(gen), pos_dist(gen));
            crack.length = len_dist(gen);
            crack.angle = ang_dist(gen);
            crack.resistance = 50e6 / std::sqrt(PI * crack.length / 2.0);
            cracks.push_back(crack);
        }
    }
    
    void createDipoles(int n_per_crack) {
        for (size_t cid = 0; cid < cracks.size(); ++cid) {
            Crack& crack = cracks[cid];
            double spacing = crack.length / (n_per_crack + 1);
            
            Vec2 crack_dir(std::cos(crack.angle), std::sin(crack.angle));
            Vec2 normal(-std::sin(crack.angle), std::cos(crack.angle));
            
            for (int i = 0; i < n_per_crack; ++i) {
                double s = (i + 1) * spacing - crack.length / 2.0;
                Vec2 base = crack.center + crack_dir * s;
                double sep = 0.1 * b;
                
                Dislocation pos_disl;
                pos_disl.pos = base + normal * (sep/2);
                pos_disl.b = normal * b;
                pos_disl.crack_id = cid;
                pos_disl.is_positive = true;
                
                Dislocation neg_disl;
                neg_disl.pos = base - normal * (sep/2);
                neg_disl.b = normal * (-b);
                neg_disl.crack_id = cid;
                neg_disl.is_positive = false;
                
                crack.dipole_indices.push_back(dislocations.size() / 2);
                dislocations.push_back(pos_disl);
                dislocations.push_back(neg_disl);
            }
        }
    }
    
    Stress stressFromDislocation(const Vec2& r, const Dislocation& disl) {
        Vec2 rel = r - disl.pos;
        double x = rel.x, y = rel.y;
        double r2 = x*x + y*y;
        
        if (r2 < 0.25*b*b) return Stress(0, 0, 0);
        
        double D = G / (2.0 * PI * (1.0 - nu));
        double r4 = r2 * r2;
        double bx = disl.b.x, by = disl.b.y;
        
        Stress sigma;
        sigma.sxx = -D * by * y * (3*x*x + y*y) / r4 + 
                     D * bx * x * (x*x - y*y) / r4;
        sigma.syy = D * by * y * (x*x - y*y) / r4 - 
                     D * bx * x * (x*x + 3*y*y) / r4;
        sigma.sxy = D * by * x * (x*x - y*y) / r4 - 
                     D * bx * y * (x*x - y*y) / r4;
        
        return sigma;
    }
    
    Stress totalStressAt(const Vec2& point) {
        Stress total = external_stress;
        for (const auto& disl : dislocations) {
            total = total + stressFromDislocation(point, disl);
        }
        return total;
    }
    
    Vec2 peachKoehlerForce(size_t idx) {
        const Dislocation& disl = dislocations[idx];
        Stress sigma = totalStressAt(disl.pos);
        
        double sigma_b_x = sigma.sxx * disl.b.x + sigma.sxy * disl.b.y;
        double sigma_b_y = sigma.sxy * disl.b.x + sigma.syy * disl.b.y;
        
        return Vec2(sigma_b_y, -sigma_b_x);
    }
    
    void solveEquilibrium() {
        double damping = 0.1;
        int max_iter = 500;
        double tol = 1e-6 * G * b;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            double max_force = 0;
            std::vector<Vec2> forces(dislocations.size());
            
            for (size_t i = 0; i < dislocations.size(); ++i) {
                forces[i] = peachKoehlerForce(i);
                max_force = std::max(max_force, forces[i].norm());
            }
            
            if (max_force < tol) {
                std::cout << "  Converged in " << iter << " iterations\n";
                return;
            }
            
            double dt = 0.01 * b / (max_force / damping + 1e-10);
            for (size_t i = 0; i < dislocations.size(); ++i) {
                dislocations[i].pos = dislocations[i].pos + forces[i] * (dt / damping);
            }
        }
        std::cout << "  Warning: max iterations reached\n";
    }
    
    bool checkPropagation(size_t crack_id) {
        const Crack& crack = cracks[crack_id];
        if (crack.dipole_indices.empty()) return false;
        
        int tip_dipole = crack.dipole_indices.back();
        int tip_idx = tip_dipole * 2;
        
        Vec2 force = peachKoehlerForce(tip_idx);
        Vec2 crack_dir(std::cos(crack.angle), std::sin(crack.angle));
        double force_mag = force.dot(crack_dir);
        
        return (force_mag > crack.resistance);
    }
    
    void propagateCrack(size_t crack_id, double extension) {
        Crack& crack = cracks[crack_id];
        double old_len = crack.length;
        crack.length += extension;
        crack.resistance = 50e6 / std::sqrt(PI * crack.length / 2.0);
        
        Vec2 crack_dir(std::cos(crack.angle), std::sin(crack.angle));
        Vec2 normal(-std::sin(crack.angle), std::cos(crack.angle));
        Vec2 new_pos = crack.center + crack_dir * (crack.length / 2);
        
        double sep = 0.1 * b;
        
        Dislocation pos_disl, neg_disl;
        pos_disl.pos = new_pos + normal * (sep/2);
        pos_disl.b = normal * b;
        pos_disl.crack_id = crack_id;
        pos_disl.is_positive = true;
        
        neg_disl.pos = new_pos - normal * (sep/2);
        neg_disl.b = normal * (-b);
        neg_disl.crack_id = crack_id;
        neg_disl.is_positive = false;
        
        crack.dipole_indices.push_back(dislocations.size() / 2);
        dislocations.push_back(pos_disl);
        dislocations.push_back(neg_disl);
        
        std::cout << "  Crack " << crack_id << " grew: " 
                  << old_len*1e6 << " -> " << crack.length*1e6 << " μm\n";
    }
    
    void writeStressField(const std::string& filename, 
                         double xmin, double xmax, int nx,
                         double ymin, double ymax, int ny) {
        std::ofstream file(filename);
        file << std::scientific << std::setprecision(8);
        
        double dx = (xmax - xmin) / (nx - 1);
        double dy = (ymax - ymin) / (ny - 1);
        
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                Vec2 point(x, y);
                
                bool skip = false;
                for (const auto& disl : dislocations) {
                    if ((point - disl.pos).norm() < 2*b) {
                        skip = true;
                        break;
                    }
                }
                
                if (skip) {
                    file << x << " " << y << " 0 0 0\n";
                    continue;
                }
                
                Stress sigma = totalStressAt(point);
                file << x << " " << y << " " 
                     << sigma.sxx << " " << sigma.syy << " " << sigma.sxy << "\n";
            }
        }
        
        file << "-999 -999 -999 -999 -999\n";
        
        for (const auto& crack : cracks) {
            file << crack.center.x << " " << crack.center.y << " "
                 << crack.length << " " << crack.angle << "\n";
        }
        
        file << "-998 -998 -998 -998\n";
        
        for (const auto& disl : dislocations) {
            file << disl.pos.x << " " << disl.pos.y << " "
                 << disl.b.x << " " << disl.b.y << " "
                 << (disl.is_positive ? 1 : -1) << "\n";
        }
        
        file.close();
    }
    
    void printStats() {
        double mean_len = 0, max_len = 0, min_len = 1e10;
        for (const auto& c : cracks) {
            mean_len += c.length;
            max_len = std::max(max_len, c.length);
            min_len = std::min(min_len, c.length);
        }
        mean_len /= cracks.size();
        
        std::cout << "  Mean: " << mean_len*1e6 << " μm, Max: " << max_len*1e6 << " μm\n";
    }
    
    int getNumCracks() const { return cracks.size(); }
    int getNumDislocations() const { return dislocations.size(); }
};

int main() {
    SimpleCrackSystem system;
    
    system.initializeCracks(5, 2e-6, 5e-6, 40e-6, 12345);
    system.createDipoles(8);
    
    std::cout << "Initialized " << system.getNumCracks() << " cracks, "
              << system.getNumDislocations() << " dislocations\n\n";
    
    double sigma_min = 50e6, sigma_max = 300e6, d_sigma = 25e6;
    int n_steps = (int)((sigma_max - sigma_min) / d_sigma) + 1;
    
    for (int step = 0; step < n_steps; ++step) {
        double sigma = sigma_min + step * d_sigma;
        
        std::cout << "\n=== Step " << step << ": σ = " << sigma/1e6 << " MPa ===\n";
        
        system.setExternalStress(sigma, 0, 0);
        system.solveEquilibrium();
        
        int n_prop = 0;
        for (int i = 0; i < system.getNumCracks(); ++i) {
            if (system.checkPropagation(i)) {
                system.propagateCrack(i, 0.5e-6);
                n_prop++;
            }
        }
        
        std::cout << "  Propagations: " << n_prop << "\n";
        system.printStats();
        
        if (step % 2 == 0) {
            std::string fname = "field_" + std::to_string(step) + ".dat";
            system.writeStressField(fname, -25e-6, 25e-6, 100, -25e-6, 25e-6, 100);
            std::cout << "  Wrote " << fname << "\n";
        }
    }
    
    std::cout << "\n=== Simulation Complete ===\n";
    return 0;
}