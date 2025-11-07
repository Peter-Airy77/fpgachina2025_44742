// ================================================================
// Module: uart_tx
// Description: UART transmitter with 8N1 format (8 data bits, no parity, 1 stop bit)
// Baud rate: 115200
// ================================================================
module uart_tx(
    input               hs_clk,           // System clock (50MHz)
    input               rst_n,            // Active low reset
    // Control interface
    input               uart_tx_en,       // UART transmit enable
    input       [7:0]   uart_tx_data,     // Data to transmit
    // UART output interface
    output  reg         uart_txd,         // UART transmit data line
    // Status indicators
    output  reg         uart_tx_busy,     // Transmitter busy
    output  reg         tx_done,          // Transmission complete
    output  reg         tx_error          // Transmission error
);

// Parameters
parameter CLK_FREQ = 50000000;           // System clock frequency: 50MHz
parameter UART_BPS = 115200;             // Baud rate: 115200
localparam BAUD_CNT_MAX = CLK_FREQ / UART_BPS; // Clock cycles per bit

// State machine parameters
parameter [3:0] TX_IDLE     = 4'b0000;   // Idle state
parameter [3:0] TX_START    = 4'b0001;   // Transmit start bit
parameter [3:0] TX_DATA0    = 4'b0010;   // Transmit data bit 0
parameter [3:0] TX_DATA1    = 4'b0011;   // Transmit data bit 1
parameter [3:0] TX_DATA2    = 4'b0100;   // Transmit data bit 2
parameter [3:0] TX_DATA3    = 4'b0101;   // Transmit data bit 3
parameter [3:0] TX_DATA4    = 4'b0110;   // Transmit data bit 4
parameter [3:0] TX_DATA5    = 4'b0111;   // Transmit data bit 5
parameter [3:0] TX_DATA6    = 4'b1000;   // Transmit data bit 6
parameter [3:0] TX_DATA7    = 4'b1001;   // Transmit data bit 7
parameter [3:0] TX_STOP     = 4'b1010;   // Transmit stop bit
parameter [3:0] TX_ERROR    = 4'b1011;   // Error state

// Register declarations
reg [3:0]   current_state;    // Current state
reg [3:0]   next_state;       // Next state
reg [7:0]   tx_data_reg;      // Transmit data register
reg [15:0]  baud_cnt;         // Baud rate counter
reg         baud_tick;        // Baud rate tick pulse
reg [15:0]  error_timeout;    // Error timeout counter
reg         tx_en_d0;         // Transmit enable delay
reg         tx_en_d1;         // Transmit enable delay

//*****************************************************
//**                    main code
//*****************************************************

// Detect positive edge of transmit enable
always @(posedge hs_clk or negedge rst_n) begin
    if(!rst_n) begin
        tx_en_d0 <= 1'b0;
        tx_en_d1 <= 1'b0;
    end
    else begin
        tx_en_d0 <= uart_tx_en;
        tx_en_d1 <= tx_en_d0;
    end
end

wire tx_en_posedge = !tx_en_d1 && tx_en_d0;  // Positive edge of tx_en

// Baud rate counter
always @(posedge hs_clk or negedge rst_n) begin
    if(!rst_n) begin
        baud_cnt <= 16'd0;
        baud_tick <= 1'b0;
    end
    else begin
        if(current_state != TX_IDLE && current_state != TX_ERROR) begin
            if(baud_cnt < BAUD_CNT_MAX - 1) begin
                baud_cnt <= baud_cnt + 16'd1;
                baud_tick <= 1'b0;
            end
            else begin
                baud_cnt <= 16'd0;
                baud_tick <= 1'b1;
            end
        end
        else begin
            baud_cnt <= 16'd0;
            baud_tick <= 1'b0;
        end
    end
end

// Error timeout counter
always @(posedge hs_clk or negedge rst_n) begin
    if(!rst_n) begin
        error_timeout <= 16'd0;
        tx_error <= 1'b0;
    end
    else begin
        if(current_state != TX_IDLE && current_state != TX_ERROR) begin
            if(error_timeout < 16'd100000) begin  // ~2ms timeout
                error_timeout <= error_timeout + 16'd1;
                tx_error <= 1'b0;
            end
            else begin
                tx_error <= 1'b1;  // Timeout error
            end
        end
        else begin
            error_timeout <= 16'd0;
            tx_error <= 1'b0;
        end
    end
end

// State register
always @(posedge hs_clk or negedge rst_n) begin
    if(!rst_n)
        current_state <= TX_IDLE;
    else if(tx_error)
        current_state <= TX_ERROR;
    else
        current_state <= next_state;
end

// State transition logic
always @(*) begin
    case(current_state)
        TX_IDLE: begin
            if(tx_en_posedge && !uart_tx_busy)
                next_state = TX_START;
            else
                next_state = TX_IDLE;
        end
        
        TX_START: begin
            if(baud_tick)
                next_state = TX_DATA0;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_START;
        end
        
        TX_DATA0: begin
            if(baud_tick)
                next_state = TX_DATA1;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_DATA0;
        end
        
        TX_DATA1: begin
            if(baud_tick)
                next_state = TX_DATA2;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_DATA1;
        end
        
        TX_DATA2: begin
            if(baud_tick)
                next_state = TX_DATA3;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_DATA2;
        end
        
        TX_DATA3: begin
            if(baud_tick)
                next_state = TX_DATA4;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_DATA3;
        end
        
        TX_DATA4: begin
            if(baud_tick)
                next_state = TX_DATA5;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_DATA4;
        end
        
        TX_DATA5: begin
            if(baud_tick)
                next_state = TX_DATA6;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_DATA5;
        end
        
        TX_DATA6: begin
            if(baud_tick)
                next_state = TX_DATA7;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_DATA6;
        end
        
        TX_DATA7: begin
            if(baud_tick)
                next_state = TX_STOP;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_DATA7;
        end
        
        TX_STOP: begin
            if(baud_tick)
                next_state = TX_IDLE;
            else if(tx_error)
                next_state = TX_ERROR;
            else
                next_state = TX_STOP;
        end
        
        TX_ERROR: begin
            // Requires external reset to recover
            next_state = TX_ERROR;
        end
        
        default: next_state = TX_IDLE;
    endcase
end

// Transmit data handling
always @(posedge hs_clk or negedge rst_n) begin
    if(!rst_n) begin
        tx_data_reg <= 8'b0;
        uart_tx_busy <= 1'b0;
        tx_done <= 1'b0;
    end
    else begin
        // Latch transmit data
        if(tx_en_posedge && current_state == TX_IDLE)
            tx_data_reg <= uart_tx_data;
        
        // Busy status indicator
        uart_tx_busy <= (current_state != TX_IDLE && current_state != TX_ERROR);
        
        // Transmission complete (when stop bit completes)
        if(current_state == TX_STOP && baud_tick)
            tx_done <= 1'b1;
        else
            tx_done <= 1'b0;
    end
end

// UART transmit data output
always @(posedge hs_clk or negedge rst_n) begin
    if(!rst_n)
        uart_txd <= 1'b1;
    else if(tx_error)
        uart_txd <= 1'b1;  // Idle high on error
    else begin
        case(current_state)
            TX_IDLE:   uart_txd <= 1'b1;
            TX_START:  uart_txd <= 1'b0;
            TX_DATA0:  uart_txd <= tx_data_reg[0];
            TX_DATA1:  uart_txd <= tx_data_reg[1];
            TX_DATA2:  uart_txd <= tx_data_reg[2];
            TX_DATA3:  uart_txd <= tx_data_reg[3];
            TX_DATA4:  uart_txd <= tx_data_reg[4];
            TX_DATA5:  uart_txd <= tx_data_reg[5];
            TX_DATA6:  uart_txd <= tx_data_reg[6];
            TX_DATA7:  uart_txd <= tx_data_reg[7];
            TX_STOP:   uart_txd <= 1'b1;
            TX_ERROR:  uart_txd <= 1'b1;
            default:   uart_txd <= 1'b1;
        endcase
    end
end

endmodule

