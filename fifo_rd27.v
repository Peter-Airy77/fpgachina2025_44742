// ================================================================
// Module: fifo_rd
// Description: FIFO read controller with UART interface
// Reads 256 bytes from FIFO and transmits via UART
// ================================================================
module fifo_rd(
    // System clock
    input               rd_clk,           // Clock signal (50MHz)
    input               rst_n,            // Active low reset
    // FIFO interface
    input               rd_rst_busy,      // Read reset busy
    input       [7:0]   fifo_rd_data,     // Data read from FIFO
    input               full,             // FIFO full
    input               almost_empty,     // FIFO almost empty
    input               empty,            // FIFO empty
    input               uart_tx_busy,     // UART transmitter busy
    // Scan control interface
    input               frame_done,       // Frame done signal
    // Output interface
    output reg          fifo_rd_en,       // FIFO read enable
    output reg          uart_tx_en,       // UART transmit enable
    output reg  [7:0]   uart_tx_data,     // UART transmit data
    output reg          frame_complete,   // Frame transmission complete
    output reg  [8:0]   read_count,       // Read counter (0-256, needs 9 bits)
    output reg          transmission_error // Transmission error indicator
);

// State machine parameters
parameter [2:0] IDLE           = 3'b000;  // Idle state
parameter [2:0] CHECK_FIFO     = 3'b001;  // Check FIFO status
parameter [2:0] READ_FIFO      = 3'b010;  // Read from FIFO
parameter [2:0] WAIT_UART      = 3'b011;  // Wait for UART ready
parameter [2:0] SEND_UART      = 3'b100;  // Send data to UART
parameter [2:0] WAIT_FINISH    = 3'b101;  // Wait for transmission to finish
parameter [2:0] FRAME_COMPLETE = 3'b110;  // Frame transmission complete

// Register declarations
reg [2:0] current_state;      // Current state
reg [2:0] next_state;         // Next state
reg       full_d0;
reg       full_d1;
reg       frame_done_d0;
reg       frame_done_d1;
reg       frame_active;       // Frame active flag
reg [15:0] timeout_counter;   // Timeout counter

//*****************************************************
//**                    main code
//*****************************************************

// Synchronize full signal to read clock domain
always @(posedge rd_clk or negedge rst_n) begin
    if(!rst_n) begin
        full_d0 <= 1'b0;
        full_d1 <= 1'b0;
    end
    else begin
        full_d0 <= full;
        full_d1 <= full_d0;
    end
end

// Synchronize frame_done signal
always @(posedge rd_clk or negedge rst_n) begin
    if(!rst_n) begin
        frame_done_d0 <= 1'b0;
        frame_done_d1 <= 1'b0;
    end
    else begin
        frame_done_d0 <= frame_done;
        frame_done_d1 <= frame_done_d0;
    end
end

// Frame active flag and read counter logic (combined to avoid multiple drivers)
always @(posedge rd_clk or negedge rst_n) begin
    if(!rst_n) begin
        frame_active <= 1'b0;
        read_count <= 9'b0;
        frame_complete <= 1'b0;
        transmission_error <= 1'b0;
    end
    else begin
        // Frame active control
        if(timeout_counter > 16'd50000) begin  // Timeout condition
            frame_active <= 1'b0;
            transmission_error <= 1'b1;
        end
        else if(full_d1 && !frame_active) begin  // FIFO full and not currently processing
            frame_active <= 1'b1;
            transmission_error <= 1'b0;
        end
        else if(frame_complete) begin            // Transmission complete
            frame_active <= 1'b0;
        end
        
        // Read counter control
        if(current_state == READ_FIFO && fifo_rd_en) begin
            read_count <= read_count + 1'b1;
            frame_complete <= 1'b0;
        end
        else if(read_count == 9'd256) begin
            read_count <= 9'b0;
            frame_complete <= 1'b1;      // All 256 data transmitted
            transmission_error <= 1'b0;
        end
        else if(frame_done_d1 && read_count != 9'd256 && read_count != 9'b0) begin
            transmission_error <= 1'b1;  // Frame done but data incomplete
        end
        else begin
            frame_complete <= 1'b0;
        end
    end
end

// Timeout counter
always @(posedge rd_clk or negedge rst_n) begin
    if(!rst_n)
        timeout_counter <= 16'b0;
    else if(frame_active && !frame_complete)
        timeout_counter <= timeout_counter + 1'b1;
    else
        timeout_counter <= 16'b0;
end

// State register
always @(posedge rd_clk or negedge rst_n) begin
    if(!rst_n)
        current_state <= IDLE;
    else if(rd_rst_busy)
        current_state <= IDLE;
    else
        current_state <= next_state;
end

// State transition logic
always @(*) begin
    case(current_state)
        IDLE: begin
            if(!rd_rst_busy && frame_active && read_count < 9'd256)
                next_state = CHECK_FIFO;
            else
                next_state = IDLE;
        end
        
        CHECK_FIFO: begin
            if(!empty && read_count < 9'd256)
                next_state = READ_FIFO;
            else if(read_count == 9'd256)
                next_state = FRAME_COMPLETE;
            else if(almost_empty && frame_done_d1)
                next_state = IDLE;
            else
                next_state = CHECK_FIFO;
        end
        
        READ_FIFO: begin
            next_state = WAIT_UART;
        end
        
        WAIT_UART: begin
            if(!uart_tx_busy)
                next_state = SEND_UART;
            else
                next_state = WAIT_UART;
        end
        
        SEND_UART: begin
            next_state = WAIT_FINISH;
        end
        
        WAIT_FINISH: begin
            if(uart_tx_busy)
                next_state = CHECK_FIFO;
            else if(read_count == 9'd256)
                next_state = FRAME_COMPLETE;
            else
                next_state = WAIT_FINISH;
        end
        
        FRAME_COMPLETE: begin
            next_state = IDLE;
        end
        
        default: next_state = IDLE;
    endcase
end

// Output logic
always @(posedge rd_clk or negedge rst_n) begin
    if(!rst_n) begin
        fifo_rd_en   <= 1'b0;
        uart_tx_en   <= 1'b0;
        uart_tx_data <= 8'b0;
    end
    else begin
        case(current_state)
            IDLE: begin
                fifo_rd_en   <= 1'b0;
                uart_tx_en   <= 1'b0;
                uart_tx_data <= 8'b0;
            end
            
            CHECK_FIFO: begin
                fifo_rd_en   <= 1'b0;
                uart_tx_en   <= 1'b0;
            end
            
            READ_FIFO: begin
                fifo_rd_en   <= 1'b1;           // Enable FIFO read
                uart_tx_en   <= 1'b0;
            end
            
            WAIT_UART: begin
                fifo_rd_en   <= 1'b0;
                uart_tx_data <= fifo_rd_data;   // Latch FIFO data after read
                uart_tx_en   <= 1'b0;
            end
            
            SEND_UART: begin
                fifo_rd_en <= 1'b0;
                uart_tx_en <= 1'b1;             // Start UART transmission
            end
            
            WAIT_FINISH: begin
                fifo_rd_en <= 1'b0;
                uart_tx_en <= 1'b0;
            end
            
            FRAME_COMPLETE: begin
                fifo_rd_en <= 1'b0;
                uart_tx_en <= 1'b0;
            end
            
            default: begin
                fifo_rd_en   <= 1'b0;
                uart_tx_en   <= 1'b0;
                uart_tx_data <= 8'b0;
            end
        endcase
    end
end

endmodule

