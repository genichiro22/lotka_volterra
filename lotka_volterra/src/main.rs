use std::fs::File;
use std::io::prelude::*;

// 1. Define the Species struct
#[derive(Clone, Copy)]
struct Species {
    population: f64,
    diffusion_coefficient: f64,
}

// 2. Define the Parameters struct
#[derive(Clone, Copy)]
struct Parameters {
    alpha: [f64; 2],
    beta: [[f64; 2]; 2],
    gamma: [f64; 2],
    delta: [[f64; 2]; 2],
    psi: [f64; 2],
    kappa: [f64; 2],
    lambda: [f64; 2],
    xi: [f64; 2],
}

// 3. Define the LotkaVolterra struct
struct LotkaVolterra {
    p1: Species,
    p2: Species,
    q1: Species,
    q2: Species,
    params: Parameters,
    grid_size: usize,
    dx: f64,
    dt: f64,
}

impl LotkaVolterra {
    // 4. Initialize the LotkaVolterra struct
    fn new(
        grid_size: usize,
        dx: f64,
        dt: f64,
        p1_init: &Vec<Vec<f64>>,
        p2_init: &Vec<Vec<f64>>,
        q1_init: &Vec<Vec<f64>>,
        q2_init: &Vec<Vec<f64>>,
        params: Parameters,
    ) -> Self {
        let p1 = p1_init
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&pop| Species {
                        population: pop,
                        diffusion_coefficient: params.alpha[0],
                    })
                    .collect()
            })
            .collect();
        let p2 = p2_init
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&pop| Species {
                        population: pop,
                        diffusion_coefficient: params.alpha[1],
                    })
                    .collect()
            })
            .collect();
        let q1 = q1_init
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&pop| Species {
                        population: pop,
                        diffusion_coefficient: params.gamma[0],
                    })
                    .collect()
            })
            .collect();
        let q2 = q2_init
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&pop| Species {
                        population: pop,
                        diffusion_coefficient: params.gamma[1],
                    })
                    .collect()
            })
            .collect();

        LotkaVolterra {
            p1,
            p2,
            q1,
            q2,
            params,
            grid_size,
            dx,
            dt,
        }
    }  

    // 5. Compute the Laplacian for each species
    fn laplacian(&self, species: &Vec<Vec<Species>>, x: usize, y: usize) -> f64 {
        let left = species[x.wrapping_sub(1)][y].population;
        let right = species[(x + 1) % self.grid_size][y].population;
        let up = species[x][y.wrapping_sub(1)].population;
        let down = species[x][(y + 1) % self.grid_size].population;
        let center = species[x][y].population;
        let laplacian = (left + right + up + down - 4.0 * center) / (self.dx * self.dx);
        laplacian
    }

    // 6. Compute the derivatives for each species
    fn derivatives(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut d_p1 = vec![vec![0.0; self.grid_size]; self.grid_size];
        let mut d_p2 = vec![vec![0.0; self.grid_size]; self.grid_size];
        let mut d_q1 = vec![vec![0.0; self.grid_size]; self.grid_size];
        let mut d_q2 = vec![vec![0.0; self.grid_size]; self.grid_size];
    
        for i in 0..self.grid_size {
            for j in 0..self.grid_size {
                let p1 = self.p1[i][j].population;
                let p2 = self.p2[i][j].population;
                let q1 = self.q1[i][j].population;
                let q2 = self.q2[i][j].population;
    
                let laplacian_p1 = self.laplacian(&self.p1, i, j);
                let laplacian_p2 = self.laplacian(&self.p2, i, j);
                let laplacian_q1 = self.laplacian(&self.q1, i, j);
                let laplacian_q2 = self.laplacian(&self.q2, i, j);
    
                d_p1[i][j] = self.params.alpha[0] * p1 * (1.0 - (p1 + self.params.psi[0] * p2) / self.params.kappa[0])
                    - self.params.beta[0][0] * p1 * q1
                    - self.params.beta[0][1] * p1 * q2
                    + self.p1[i][j].diffusion_coefficient * laplacian_p1;
    
                d_p2[i][j] = self.params.alpha[1] * p2 * (1.0 - (p2 + self.params.psi[1] * p1) / self.params.kappa[1])
                    - self.params.beta[1][0] * p2 * q1
                    - self.params.beta[1][1] * p2 * q2
                    + self.p2[i][j].diffusion_coefficient * laplacian_p2;
    
                d_q1[i][j] = self.params.delta[0][0] * p1 * q1
                    + self.params.delta[0][1] * p2 * q1
                    - self.params.gamma[0] * q1 * (1.0 - (q1 + self.params.xi[0] * q2) / self.params.lambda[0])
                    + self.q1[i][j].diffusion_coefficient * laplacian_q1;
    
                d_q2[i][j] = self.params.delta[1][0] * p1 * q2
                    + self.params.delta[1][1] * p2 * q2
                    - self.params.gamma[1] * q2 * (1.0 - (q2 + self.params.xi[1] * q1) / self.params.lambda[1])
                    + self.q2[i][j].diffusion_coefficient * laplacian_q2;
            }
        }
    
        (d_p1, d_p2, d_q1, d_q2)
    }

    // 7. Integrate the system using a numerical integration technique
    fn integrate(&mut self) {
        let (d_p1, d_p2, d_q1, d_q2) = self.derivatives();
        for i in 0..self.grid_size {
            for j in 0..self.grid_size {
                self.p1[i][j].population += self.dt * d_p1[i][j];
                self.p2[i][j].population += self.dt * d_p2[i][j];
                self.q1[i][j].population += self.dt * d_q1[i][j];
                self.q2[i][j].population += self.dt * d_q2[i][j];
            }
        }
    }
    
    // 8. Run the simulation for a specified number of time steps
    pub fn run_simulation(&mut self, steps: usize) {
        for _ in 0..steps {
            self.integrate();
        }
    }
    
    // 9. Export the simulation results or visualize them
    pub fn export_results(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
    
        writeln!(file, "P1")?;
        for row in &self.p1 {
            let row_str: Vec<String> = row.iter().map(|cell| cell.population.to_string()).collect();
            writeln!(file, "{}", row_str.join(","))?;
        }
    
        writeln!(file, "\nP2")?;
        for row in &self.p2 {
            let row_str: Vec<String> = row.iter().map(|cell| cell.population.to_string()).collect();
            writeln!(file, "{}", row_str.join(","))?;
        }
    
        writeln!(file, "\nQ1")?;
        for row in &self.q1 {
            let row_str: Vec<String> = row.iter().map(|cell| cell.population.to_string()).collect();
            writeln!(file, "{}", row_str.join(","))?;
        }
    
        writeln!(file, "\nQ2")?;
        for row in &self.q2 {
            let row_str: Vec<String> = row.iter().map(|cell| cell.population.to_string()).collect();
            writeln!(file, "{}", row_str.join(","))?;
        }
    
        Ok(())
    }
}
fn main() {
    // Initialize parameters
    let params = Parameters {
        alpha: [0.1, 0.1],
        beta: [[0.02, 0.02], [0.02, 0.02]],
        gamma: [0.3, 0.3],
        delta: [[0.01, 0.01], [0.01, 0.01]],
        psi: [0.1, 0.1],
        kappa: [100.0, 100.0],
        lambda: [100.0, 100.0],
        xi: [0.1, 0.1],
    };

    // Initialize population matrices
    let p1_init = vec![vec![50.0; 50]; 50];
    let p2_init = vec![vec![50.0; 50]; 50];
    let q1_init = vec![vec![100.0; 50]; 50];
    let q2_init = vec![vec![100.0; 50]; 50];

    // Create a LotkaVolterra simulation
    let mut simulation = LotkaVolterra::new(50, 1.0, 0.01, &p1_init, &p2_init, &q1_init, &q2_init, params);

    // Run the simulation for a specified number of time steps
    simulation.run_simulation(1000);

    // Export the results to a CSV file
    simulation.export_results("output.csv").unwrap();

    println!("Simulation results exported to output.csv");
}