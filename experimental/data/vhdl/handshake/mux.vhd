library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity mux is
  generic (
    INPUTS        : integer;
    BITWIDTH      : integer;
    COND_BITWIDTH : integer
  );
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    condition  : in std_logic_vector(COND_BITWIDTH - 1 downto 0);
    ins        : in data_array(INPUTS - 2 downto 0)(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic_vector(INPUTS - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    ins_ready  : out std_logic_vector(INPUTS - 1 downto 0);
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
end mux;

architecture arch of mux is

  signal tehb_data_in : std_logic_vector(BITWIDTH - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;

begin
  process (ins, ins_valid, outs_ready, condition, tehb_ready)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    for I in INPUTS - 2 downto 0 loop
      if (unsigned(condition) = to_unsigned(I, condition'length) and ins_valid(0) = '1' and ins_valid(I + 1) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := '1';
      end if;

      if ((unsigned(condition) = to_unsigned(I, condition'length) and ins_valid(0) = '1' and tehb_ready = '1' and ins_valid(I + 1) = '1') or ins_valid(I + 1) = '0') then
        ins_ready(I + 1) <= '1';
      else
        ins_ready(I + 1) <= '0';
      end if;
    end loop;

    if (ins_valid(0) = '0' or (tmp_valid_out = '1' and tehb_ready = '1')) then
      ins_ready(0) <= '1';
    else
      ins_ready(0) <= '0';
    end if;

    tehb_data_in <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    tehb_pvalid  <= tmp_valid_out;
  end process;
  tehb1 : entity work.TEHB(arch) generic map (BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready,
      ins        => tehb_data_in,
      outs       => outs
    );
end arch;
